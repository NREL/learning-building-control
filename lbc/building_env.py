"""
This environment is intended to help validate and reproduce results
by directly utilizing the pytorch functions in building_dynamics.py to run
against a Gym API whereby traditional RL algorithms (e.g., PPO) can be tested.
The implementation is suboptimal in that it assumes scenario batching and
converts numpy arrays to (batched) torch tensors, and back again, when
evolving the dynamics.
"""

import gym
import numpy as np
import torch

import lbc.dynamics as dyn

from gym import spaces

from lbc.scenario import Scenario
from lbc.utils import to_torch
from lbc.costs import stage_cost

FILTER_UPDATE_CNT = 2
STEPS_PER_HOUR = 12
ZONE_TEMP_BOUNDS = (16., 40.)
OUTDOOR_TEMP_BOUNDS = (0., 56.)
Q_SOLAR_BOUNDS = (0., 12.)
PRICE_BOUNDS = (0, 10.)
POWER_LIMIT_BOUNDS = (0., 25.)
REWARD_SCALING_FACTOR = 0.001


class BuildingControlEnv(gym.Env):
    """ Five zone building control environment.

    This environment leverages existing building dynamics from dynamics.py.
    """

    def __init__(self,
                 scenario: Scenario = Scenario(),
                 training: bool = True):
        """ Initialize building control environment.

        Args:
          scenario: a instance of the Scenario class, which contains
            information regarding the control episode, building model,
            demand response configures and more.
          training: a Boolean variable indicates this environment is used for
            training or testing.
        """

        self.scenario = scenario
        self.training = training
        self.num_zone = len(scenario.zone_model['x_k'])

        # Load exogenous dataset (include all days).
        if self.training:
            self.total_day_num = len(self.scenario.train_dates)
        else:
            self.total_day_num = len(self.scenario.test_dates)
        self.exogenous_data = scenario.make_batch(
            self.total_day_num, training=training)

        # Define Gym required variables
        # See "get_obs" functions for details.
        dr_type = self.scenario.dr_program.program_type
        if dr_type == 'TOU':
            dim_obs = 14
        elif dr_type == 'PC':
            dim_obs = 26
        elif dr_type == 'RTP':
            dim_obs = 30
        self.scalar_obs_upper = np.array([np.inf] * dim_obs)
        self.scalar_obs_lower = np.array([-np.inf] * dim_obs)
        self.observation_space = spaces.Box(self.scalar_obs_lower,
                                            self.scalar_obs_upper,
                                            dtype=np.float)

        self.action_upper = np.array([1.0] * (self.num_zone + 1))
        self.action_lower = np.array([-1.0] * (self.num_zone + 1))
        self.action_space = spaces.Box(self.action_lower,
                                       self.action_upper,
                                       dtype=np.float32)

        self.x = torch.zeros(self.num_zone)  # State.

        self.temp_oa = None
        self.zone_temp = None
        self.q_solar = None
        self.q_int = None  # TODO @xzhang2 Check if this one is used?
        self.energy_price = None
        self.comfort_max = None
        self.comfort_min = None
        self.predicted_energy_price = None
        self.price_diff = None
        self.step_idx = 0

    def reset(self, day_idx=None):
        """ Resetting this control episode by choosing one day.

        Args:
          day_idx: integer (Optional). Day index for initialize the control
            episode. Usually will specify when testing under a specific day
            is needed.
        """

        if day_idx is None:
            day_idx = np.random.randint(self.total_day_num)

        # Populate the exogenous data variable needed in this episode.
        self.temp_oa = self.exogenous_data.temp_oa[day_idx, :]  # [287]
        self.q_solar = self.exogenous_data.q_solar[day_idx, :, :]  # [287, 5]
        self.zone_temp = self.exogenous_data.zone_temp[day_idx, :]
        self.energy_price = self.exogenous_data.energy_price[day_idx, :]
        self.predicted_energy_price = \
            self.exogenous_data.predicted_energy_price[day_idx, :]
        self.price_diff = (self.energy_price
                           - self.predicted_energy_price).numpy()
        self.comfort_max = self.exogenous_data.comfort_max[day_idx, :]
        self.comfort_min = self.exogenous_data.comfort_min[day_idx, :]

        self.step_idx = 0
        self.filter_update()

        obs = self.get_obs()

        return obs

    def filter_update(self):
        """ Use filter update to obtain the state value, used at the initial
        step.
        """

        action = self.scenario.action_init
        if type(action) != torch.Tensor:
            action = torch.tensor(action)

        u = dyn.build_u_vector(
            model=self.scenario.zone_model,
            zone_temp=self.zone_temp.reshape(-1, self.num_zone),
            temp_oa=self.temp_oa[self.step_idx:
                                 self.step_idx+1].reshape(-1, 1),
            q_solar=self.q_solar[self.step_idx, :].reshape(-1, self.num_zone),
            action=action.reshape(-1, self.num_zone + 1)
        )

        for _ in range(FILTER_UPDATE_CNT):
            self.x = dyn.filter_update(
                x=self.x, u=u, zone_temp=self.zone_temp,
                model=self.scenario.zone_model)

    def step(self, action):
        """ Advance the environment for one step.
        """

        action = np.array([(x + 1) / 2 for x in action])  # to [0, 1] range

        action = to_torch(self.scenario.action_min + action * (
            self.scenario.action_max - self.scenario.action_min))
        clipped_action = to_torch(np.clip(action,
                                          self.scenario.action_min,
                                          self.scenario.action_max))

        # Unsqueeze to put actions into a batch format, which is needed by
        # the dynamics.
        action = action.unsqueeze(0)
        clipped_action = clipped_action.unsqueeze(0)

        action_min = to_torch(self.scenario.action_min, batch_size=1)
        action_max = to_torch(self.scenario.action_max, batch_size=1)
        action = action.reshape(*action_min.shape)

        if self.scenario.dr_program.program_type == 'PC':
            pc_limit = self.scenario.dr_program.power_limit.values[
                self.step_idx][0] * torch.ones(1)
            pc_penalty = self.scenario.dr_program.pc_penalty
        else:
            pc_limit = None
            pc_penalty = None

        comfort_min = self.comfort_min[self.step_idx].unsqueeze(-1)
        comfort_max = self.comfort_max[self.step_idx].unsqueeze(-1)

        # TODO: Is it worth considering the following question:
        # Do we calculate cost first or move the dynamic first or it does not
        # matter?
        costs = stage_cost(
            zone_model=self.scenario.zone_model,
            action=action,
            clipped_action=clipped_action,
            temp_oa=self.temp_oa[self.step_idx: self.step_idx + 1],
            zone_temp=self.zone_temp,
            energy_price=self.energy_price[self.step_idx],
            action_penalty=(self.scenario.action_penalty
                            if self.training else 0.),
            action_min=action_min,
            action_max=action_max,
            comfort_penalty=self.scenario.comfort_penalty,
            comfort_min=comfort_min,
            comfort_max=comfort_max,
            pc_penalty=pc_penalty,
            pc_limit=pc_limit)

        # Evolve the dynamics
        self.x, self.zone_temp, _ = dyn.dynamics(
            x=self.x, zone_temp=self.zone_temp.reshape(-1, self.num_zone),
            action=clipped_action.reshape(-1, self.num_zone + 1),
            temp_oa=self.temp_oa[self.step_idx: self.step_idx +
                                 1].reshape(-1, 1),
            q_solar=self.q_solar[self.step_idx: self.step_idx + 1,
                                 :].reshape(-1, self.num_zone),
            model=self.scenario.zone_model
        )

        self.x = self.x.squeeze()
        self.zone_temp = self.zone_temp.squeeze()

        # Down scaling reward as it relates to the value function clipping for
        # algorithms like PPO.
        reward = -costs.total_cost.numpy()[0] * REWARD_SCALING_FACTOR

        obs = self.get_obs()

        done = False
        self.step_idx += 1
        if self.step_idx >= len(self.temp_oa):
            done = True

        return obs, reward, done, {}

    def get_obs(self):
        """ Constructs the observation vector.

        Depending the type of the DR program, the observation is a little
        different:

        Basic observation (All DR program cases have) includes:
        - Indoor temperatures of five zones [5].
        - Current step index: Step index [1].
        - Trignometric representation of time [2].
        - Outdoor env: mean temperature/q_solar for the past one hour [1 + 5].


        TOU Program: No additional info is included. Price info also not
          included, since TOU price is fixed.
          Total observation dimension is 14.

        PC Program:
          - Power limit for the next one hour [12].
          Total observation dimension is 26.

        RTP Program:
          - Predicted energy price for the next hour [12].
          - Energy price and predicted price differences for the past four
            hours, i.e., RTP - DAP [4].
          Total observation dimension is 30.

        Returns:
          obs: Numpy array. The observation array.
        """

        normalized_zone_temp = self.normalize(self.zone_temp.tolist(),
                                              ZONE_TEMP_BOUNDS)

        degree = self.step_idx / 288.0 * 2 * np.pi
        trignometric_time = [np.sin(degree), np.cos(degree)]

        mean_temp_oa = np.mean(self.temp_oa.tolist()[
                               max(0, self.step_idx - STEPS_PER_HOUR):
                               self.step_idx + 1])
        normalized_temp_oa = self.normalize(mean_temp_oa, OUTDOOR_TEMP_BOUNDS)
        mean_q_solar = np.mean(self.q_solar.numpy()[
                               max(0, self.step_idx - STEPS_PER_HOUR):
                               self.step_idx + 1, :], axis=0)
        normalized_q_solar = self.normalize(mean_q_solar, Q_SOLAR_BOUNDS)

        obs = (normalized_zone_temp + [self.step_idx / 288.0]
               + trignometric_time + [normalized_temp_oa] + normalized_q_solar)

        if self.scenario.dr_program.program_type == 'PC':
            power_limit = self.scenario.dr_program.power_limit.values[
                self.step_idx: self.step_idx + STEPS_PER_HOUR]
            normalized_power_limit = self.normalize(power_limit,
                                                    POWER_LIMIT_BOUNDS)
            normalized_power_limit = self.vector_padding(
                normalized_power_limit, STEPS_PER_HOUR)
            obs += normalized_power_limit
        elif self.scenario.dr_program.program_type == 'RTP':
            price = self.predicted_energy_price.tolist()[self.step_idx:
                                                         self.step_idx
                                                         + STEPS_PER_HOUR]
            normalized_price = self.normalize(price, PRICE_BOUNDS)
            normalized_price = self.vector_padding(normalized_price,
                                                   STEPS_PER_HOUR)

            four_hour_price_diff = []
            step_pointer = self.step_idx - STEPS_PER_HOUR
            while step_pointer >= 0 and len(four_hour_price_diff) < 4:
                four_hour_price_diff.append(self.price_diff[step_pointer])
                step_pointer -= STEPS_PER_HOUR
            four_hour_price_diff = self.vector_padding(four_hour_price_diff,
                                                       4, 0.0)
            four_hour_price_diff = self.normalize(four_hour_price_diff,
                                                  PRICE_BOUNDS)
            obs += (normalized_price + four_hour_price_diff)

        obs = np.array(obs)

        return obs

    def normalize(self, items, bounds):

        mean = np.mean(bounds)
        val_range = bounds[1] - mean

        if isinstance(items, list) or isinstance(items, np.ndarray):
            if isinstance(items, np.ndarray) and len(items.shape) == 2:
                items = items.reshape(np.multiply(*items.shape))
            result = [(x - mean) / val_range for x in items]
        elif isinstance(items, float) or isinstance(items, int):
            result = (items - mean) / val_range
        else:
            raise ValueError("Values to be normalized can only be"
                             "int/float or list/Numpy array.")
        return result

    def vector_padding(self, vector, length, val=None):
        """ Padding the vector to desired length using the provided value.
        """
        if len(vector) < length:
            padding_val = val if val is not None else vector[-1]
            vector += [padding_val] * (length - len(vector))
        return vector

    def render(self, mode='human'):
        pass


if __name__ == '__main__':

    # content below can be moved to test later.
    from lbc.demand_response import DemandResponseProgram as DRP

    OBS_DIM = {'TOU': 14, 'PC': 26, 'RTP': 30}

    drp = DRP('RTP')
    s = Scenario(dr_program=drp)
    bce = BuildingControlEnv(scenario=s)

    obs = bce.reset()
    assert len(obs) == OBS_DIM[drp.program_type], (
        "Environment observation dimension does not meet requirement:"
        "expect %d, got %d" % (OBS_DIM[drp.program_type], len(obs)))

    done = False
    reward_total = 0.0
    step_cnt = 0

    print(bce.zone_temp)
    print(obs)

    while not done:
        act = bce.action_space.sample()
        obs, r, done, info = bce.step(act)
        print(obs)
        reward_total += r
        step_cnt += 1

    print("Total step number is %d." % step_cnt)
    print("Total reward is %f." % reward_total)
