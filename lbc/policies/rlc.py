import os
import logging
import pickle5 as pickle

import numpy as np
import ray

from typing import Tuple

from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env

from lbc.building_env import STEPS_PER_HOUR, BuildingControlEnv
from lbc.building_env import ZONE_TEMP_BOUNDS
from lbc.building_env import POWER_LIMIT_BOUNDS
from lbc.building_env import PRICE_BOUNDS, Q_SOLAR_BOUNDS, OUTDOOR_TEMP_BOUNDS
from lbc.demand_response import DemandResponseProgram as DRP
from lbc.policies import Policy
from lbc.scenario import Scenario, Batch
from lbc.utils import to_torch

logger = logging.getLogger(__file__)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_CHECKPOINTS = {
    # 'PC': os.path.join(THIS_DIR, 'rlc_checkpoints',
    #                    'power_constrained/checkpoint/checkpoint'),
    # 'RTP': os.path.join(THIS_DIR, 'rlc_checkpoints',
    #                     'real_time_pricing/checkpoint/checkpoint'),
    'TOU-24': os.path.join(THIS_DIR, 'rlc_checkpoints',
                           'time_of_use/24/checkpoint/checkpoint'),
    'RTP-24': os.path.join(THIS_DIR, 'rlc_checkpoints',
                           'real_time_pricing/24/checkpoint/checkpoint'),
    'PC-24': os.path.join(THIS_DIR, 'rlc_checkpoints',
                          'power_constrained/24/checkpoint/checkpoint'),
    'TOU-12': os.path.join(THIS_DIR, 'rlc_checkpoints',
                           'time_of_use/12/checkpoint/checkpoint'),
    'RTP-12': os.path.join(THIS_DIR, 'rlc_checkpoints',
                           'real_time_pricing/12/checkpoint/checkpoint'),
    'PC-12': os.path.join(THIS_DIR, 'rlc_checkpoints',
                          'power_constrained/12/checkpoint/checkpoint'),
    'TOU-6': os.path.join(THIS_DIR, 'rlc_checkpoints',
                          'time_of_use/6/checkpoint/checkpoint'),
    'RTP-6': os.path.join(THIS_DIR, 'rlc_checkpoints',
                          'real_time_pricing/6/checkpoint/checkpoint'),
    'PC-6': os.path.join(THIS_DIR, 'rlc_checkpoints',
                         'power_constrained/6/checkpoint/checkpoint'),
    'TOU-3': os.path.join(THIS_DIR, 'rlc_checkpoints',
                          'time_of_use/3/checkpoint/checkpoint'),
    'RTP-3': os.path.join(THIS_DIR, 'rlc_checkpoints',
                          'real_time_pricing/3/checkpoint/checkpoint'),
    'PC-3': os.path.join(THIS_DIR, 'rlc_checkpoints',
                         'power_constrained/3/checkpoint/checkpoint'),
}


class RLCPolicy(Policy):

    def __init__(
        self,
        num_lookahead_steps: int = 24,
        device: str = "cpu",
        node_ip_address: str = None,
        **kwargs
    ):

        super().__init__()

        self.device = device
        self.num_lookahead_steps = num_lookahead_steps

        # Use your own LAN IP below, needed if on VPN.
        if node_ip_address is not None:
            ray.init(_node_ip_address=node_ip_address)
        else:
            ray.init()

        def get_env_creator(dr_program):
            def env_creator(config):
                drp = DRP(dr_program)
                s = Scenario(dr_program=drp)
                env = BuildingControlEnv(
                    scenario=s, num_lookahead_steps=self.num_lookahead_steps)
                return env
            return env_creator

        for dr_program in ['TOU', 'RTP', 'PC']:
            env_name = ('BuildingControl' + dr_program
                        + str(self.num_lookahead_steps) + 'Env-v0')
            register_env(env_name, get_env_creator(dr_program))

        self.rl_agent = None

    def get_trained_rllib_agent(self, env_name, checkpoint):
        # Load configuration from file
        config_dir = os.path.dirname(checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        if "num_workers" in config:
            config["num_workers"] = min(1, config["num_workers"])

        cls = get_trainer_class('PPO')
        agent = cls(env=env_name, config=config)
        agent.restore(checkpoint)

        return agent

    def __call__(
        self,
        scenario: Scenario = None,
        batch: Batch = None,
        t: int = None,
        x: any = None,
        u: any = None,
        zone_temp: any = None,
        action_init: any = None,
        training: bool = True,
        **kwargs
    ) -> Tuple[any, dict]:
        """Returns the policy actions and metadata.  The keyword arguments
        will always be provided in simulate.py, it is up to the user what
        to do with them.

        Args:
            scenario:  scenario instance
            batch:  batch instance for exogenous data
            t: Integer, time index of current control step
            x: state tensor at current control step
            zone_temp:  Tensor, [batch, zone_num], zone temperature tensor
              at current control step
            action_init:  initial (or previous) action for linearization
            training:  whether we are actively training a policy or not
            **kwargs:  for customization and forward compatibility

        Returns:
            actions:  tensor of computed actions
            meta:  dictionary of metadata
        """

        dr_program = scenario.dr_program.program_type

        if self.rl_agent is None:
            env_name = ('BuildingControl' + dr_program
                        + str(self.num_lookahead_steps) + 'Env-v0')
            self.rl_agent = self.get_trained_rllib_agent(
                env_name,
                POLICY_CHECKPOINTS[dr_program + '-'
                                   + str(self.num_lookahead_steps)])

        obs = self.assemble_rl_state(scenario, x, zone_temp, t, batch,
                                     dr_program)

        bsz, _ = zone_temp.shape
        batch_action = []
        for i in range(bsz):
            rl_action = self.rl_agent.compute_action(obs[i, :])
            # Mapping actions from [-1, 1] to actual feasible range.
            rl_action = np.array([(x + 1) / 2 for x in rl_action])
            rl_action = scenario.action_min + rl_action * (
                scenario.action_max - scenario.action_min)
            batch_action.append(rl_action)
        batch_action = to_torch(batch_action)

        return batch_action, {}

    def assemble_rl_state(self, scenario, x, zone_temp, t, batch, dr_program):
        """ Assemble the inputs of the RL controller according to the
            observation space. This function is similar to the "get_obs"
            function in the building_env.py.
        """

        bce = BuildingControlEnv(num_lookahead_steps=self.num_lookahead_steps)
        zone_temp = zone_temp.numpy()
        bsz, _ = zone_temp.shape
        normalized_zone_temp = np.array(
            [bce.normalize(zone_temp[i, :], ZONE_TEMP_BOUNDS)
             for i in range(bsz)])

        temp_oa = batch.temp_oa[:, t: t + self.num_lookahead_steps].numpy()
        temp_oa = [bce.normalize(bce.vector_padding(temp_oa[i, :].tolist(),
                                                    self.num_lookahead_steps),
                                 OUTDOOR_TEMP_BOUNDS) for i in range(bsz)]
        normalized_temp_oa = np.array(temp_oa)

        q_solar = batch.q_solar[:, t: t + self.num_lookahead_steps, :].numpy()
        q_solar_batch = []
        for i in range(bsz):
            q_solar_single = q_solar[i, :, :]
            q_solar_zone = [bce.vector_padding(q_solar_single[:, j].tolist(),
                                               self.num_lookahead_steps)
                            for j in range(q_solar_single.shape[1])]
            q_solar_single = sum(q_solar_zone, [])
            q_solar_batch.append(bce.normalize(q_solar_single, Q_SOLAR_BOUNDS))

        normalized_q_solar = np.array(q_solar_batch)

        one_hot_t = [0] * 24
        one_hot_t[int(t / STEPS_PER_HOUR)] = 1
        one_hot_t_batch = np.array([one_hot_t for _ in range(bsz)])

        degree = t / 288.0 * 2 * np.pi
        trignometric_time = np.array([[np.sin(degree), np.cos(degree)]
                                      for _ in range(bsz)])

        obs = np.hstack([x.numpy(), normalized_zone_temp, normalized_temp_oa,
                         normalized_q_solar, one_hot_t_batch,
                         trignometric_time])

        if dr_program == 'PC':
            power_limit = scenario.dr_program.power_limit.values[
                t: t + self.num_lookahead_steps]
            normalized_power_limit = bce.normalize(power_limit,
                                                   POWER_LIMIT_BOUNDS)
            normalized_power_limit = bce.vector_padding(
                normalized_power_limit, self.num_lookahead_steps)
            normalized_power_limit = np.array([normalized_power_limit
                                               for _ in range(bsz)])
            obs = np.hstack([obs, normalized_power_limit])

        elif dr_program == 'RTP':

            price_forecast = batch.predicted_energy_price.numpy()[
                :, t: t + self.num_lookahead_steps]
            if price_forecast.shape[1] < self.num_lookahead_steps:
                padding = [
                    price_forecast[:, -1].reshape((-1, 1))
                    for i in range(self.num_lookahead_steps
                                   - price_forecast.shape[1])]
                padding = np.hstack(padding)
                price_forecast = np.hstack([price_forecast, padding])

            normalized_price_forecast = np.array(
                [bce.normalize(price_forecast[i, :], PRICE_BOUNDS)
                 for i in range(bsz)])

            obs = np.hstack([obs, normalized_price_forecast])

        return obs


if __name__ == '__main__':

    from lbc.simulate import simulate

    rl_policy = RLCPolicy(node_ip_address="192.168.0.16")
    drp = DRP('TOU')
    s = Scenario(dr_program=drp)

    total_loss, _, _ = simulate(policy=rl_policy, scenario=s, batch_size=30)

    print(total_loss)
    print('done')
