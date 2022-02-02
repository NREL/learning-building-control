import os
import logging
import pickle5 as pickle

import numpy as np
import ray

from typing import Tuple

from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env

from lbc.building_env import BuildingControlEnv
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
    'PC': os.path.join(THIS_DIR, 'rlc_checkpoints',
                       'power_constrained/checkpoint/checkpoint'),
    'RTP': os.path.join(THIS_DIR, 'rlc_checkpoints',
                        'real_time_pricing/checkpoint/checkpoint'),
    'TOU': os.path.join(THIS_DIR, 'rlc_checkpoints',
                        'time_of_use/checkpoint/checkpoint'),
}


class RLCPolicy(Policy):

    def __init__(
        self,
        device: str = "cpu",
        node_ip_address: str = None,
        **kwargs
    ):

        super().__init__()

        self.device = device
        self.rllib_policies = {}

        # Use your own LAN IP below, needed if on VPN.
        if node_ip_address is not None:
            ray.init(_node_ip_address=node_ip_address)
        else:
            ray.init()

        def get_env_creator(dr_program):
            def env_creator(config):
                drp = DRP(dr_program)
                s = Scenario(dr_program=drp)
                env = BuildingControlEnv(scenario=s)
                return env
            return env_creator

        for dr_program in POLICY_CHECKPOINTS.keys():
            env_name = 'BuildingControl' + dr_program + 'Env-v0'
            register_env(env_name, get_env_creator(dr_program))

            self.rllib_policies[dr_program] = self.get_trained_rllib_agent(
                env_name, POLICY_CHECKPOINTS[dr_program]
            )

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

        rl_agent = self.rllib_policies[dr_program]

        obs = self.assemble_rl_state(scenario, zone_temp, t,
                                     batch,
                                     dr_program)

        bsz, _ = zone_temp.shape
        batch_action = []
        for i in range(bsz):
            rl_action = rl_agent.compute_action(obs[i, :])
            # Mapping actions from [-1, 1] to actual feasible range.
            rl_action = np.array([(x + 1) / 2 for x in rl_action])
            rl_action = scenario.action_min + rl_action * (
                scenario.action_max - scenario.action_min)
            batch_action.append(rl_action)
        batch_action = to_torch(batch_action)

        return batch_action, {}

    def assemble_rl_state(self, scenario, zone_temp, t, batch, dr_program):
        """ Assemble the inputs of the RL controller according to the
            observation space. This function is similar to the "get_obs"
            function in the building_env.py.
        """

        bce = BuildingControlEnv()
        zone_temp = zone_temp.numpy()
        bsz, _ = zone_temp.shape
        normalized_zone_temp = np.array(
            [bce.normalize(zone_temp[i, :], ZONE_TEMP_BOUNDS)
             for i in range(bsz)])

        step_idx = np.array([t / 288.0] * bsz).reshape((bsz, 1))
        degree = t / 288.0 * 2 * np.pi
        trignometric_time = np.array([[np.sin(degree), np.cos(degree)]
                                      for _ in range(bsz)])

        mean_temp_oa = np.mean(batch.temp_oa[:, max(0, t - 12): t + 1].numpy(),
                               axis=1)
        normalized_temp_oa = np.array(bce.normalize(mean_temp_oa,
                                                    OUTDOOR_TEMP_BOUNDS))
        normalized_temp_oa = normalized_temp_oa.reshape((bsz, 1))

        mean_q_solar = np.mean(batch.q_solar[:, max(0, t - 12): t + 1,
                                             :].numpy(), axis=1)

        normalized_q_solar = np.array([bce.normalize(mean_q_solar[i, :],
                                                     Q_SOLAR_BOUNDS)
                                       for i in range(bsz)])

        obs = np.hstack([normalized_zone_temp, step_idx, trignometric_time,
                         normalized_temp_oa, normalized_q_solar])

        assert obs.shape == (bsz, 14)

        if dr_program == 'PC':
            power_limit = scenario.dr_program.power_limit.values[t: t + 12]
            normalized_power_limit = bce.normalize(power_limit,
                                                   POWER_LIMIT_BOUNDS)
            normalized_power_limit = bce.vector_padding(normalized_power_limit,
                                                        12)
            normalized_power_limit = np.array([normalized_power_limit
                                               for _ in range(bsz)])
            obs = np.hstack([obs, normalized_power_limit])

            assert obs.shape == (bsz, 26)

        elif dr_program == 'RTP':

            price_forecast = batch.predicted_energy_price.numpy()[:, t: t + 12]
            if price_forecast.shape[1] < 12:
                padding = [price_forecast[:, -1].reshape((-1, 1))
                           for i in range(12 - price_forecast.shape[1])]
                padding = np.hstack(padding)
                price_forecast = np.hstack([price_forecast, padding])

            normalized_price_forecast = np.array(
                [bce.normalize(price_forecast[i, :], PRICE_BOUNDS)
                 for i in range(bsz)])

            assert price_forecast.shape == (bsz, 12)

            price_diff = (batch.energy_price.numpy()
                          - batch.predicted_energy_price.numpy())
            batch_four_hour_price_diff = []
            step_pointer = t - 12
            while step_pointer >= 0 and len(batch_four_hour_price_diff) < 4:
                batch_four_hour_price_diff.append(
                    price_diff[:, step_pointer].reshape((-1, 1)))
                step_pointer -= 12
            if len(batch_four_hour_price_diff) == 0:
                batch_four_hour_price_diff = np.zeros((bsz, 4))
            else:
                batch_four_hour_price_diff = np.hstack(
                    batch_four_hour_price_diff)
                if batch_four_hour_price_diff.shape[1] < 4:
                    batch_four_hour_price_diff = np.hstack(
                        [batch_four_hour_price_diff,
                         np.zeros((bsz,
                                   4 - batch_four_hour_price_diff.shape[1]))]
                    )

            normalized_price_diff = np.array(
                [bce.normalize(batch_four_hour_price_diff[i, :],
                 PRICE_BOUNDS)
                 for i in range(bsz)])

            obs = np.hstack([obs, normalized_price_forecast,
                             normalized_price_diff])

            assert obs.shape == (bsz, 30)

        return obs


if __name__ == '__main__':

    from lbc.simulate import simulate

    rl_policy = RLCPolicy(node_ip_address="192.168.0.36")
    drp = DRP('RTP')
    s = Scenario(dr_program=drp)

    total_loss, _, _ = simulate(policy=rl_policy, scenario=s, batch_size=3)

    print(total_loss)
    print('done')
