import os
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch

from lbc.scenario import Scenario, Batch
from lbc.policies import Policy
from lbc.utils import to_torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "../data")


class RBCPolicy(Policy):
    """The rule based controller (RBC) implements a strategy to pre-cool
    before a peak price interval and maintain a comfortable temperature
    the rest of the day.  The strategy consists of "cooling" and
    "min-power" modes:

        Cooling: Zone flows are set to p_flow * max value, and discharge
            temp is set to p_discharge * minimum value.

        Min-power:  Zone flows are set to their minimum values and
            discharge temp is set to its maximum value.

    During the pre-cooling window, the cooling mode is activated until
    a zone goes below the comfort band, at which point the systems goes
    into min-power mode.

    At all other times, a "bang-bang" strategy is used whereby the system
    uses cooling when a zone goes above the comfort band, and min-power
    when a zone goes below.
    """

    def __init__(
        self,
        scenario: Scenario = None,
        setpoints: List[Tuple[int, float]] = None,
        band_width: float = 0.5,
        p_flow: float = None,
        p_temp: float = None,
        **kwargs
    ):

        super().__init__()

        assert p_flow >= 0. and p_flow <= 1.
        assert p_temp >= 0. and p_temp <= 1.

        self.scenario = scenario
        self.num_steps = scenario.num_episode_steps
        self.p_flow = p_flow
        self.p_temp = p_temp

        # Cooling action is parameterized by the ratio of max cooling the
        # system can provide.
        action_cooling = np.zeros_like(Scenario.action_min)
        _range = Scenario.action_max - Scenario.action_min
        action_cooling[:-1] = Scenario.action_min[:-1] + p_flow * _range[:-1]
        action_cooling[-1] = Scenario.action_max[-1] - p_temp * _range[-1]
        self.action_cooling = action_cooling

        # Minpower action minimize power consumption.
        self.action_minpower = np.zeros_like(action_cooling)
        self.action_minpower[:-1] = Scenario.action_min[:-1]
        self.action_minpower[-1] = Scenario.action_max[-1]

        self.setpoints = setpoints.copy()
        # add a dummy point to make easier
        setpoints.append((self.num_steps, setpoints[-1][1]))
        x = np.zeros((self.num_steps, 2), dtype=np.float32)
        for i in range(len(setpoints) - 1):
            icurr, tcurr = setpoints[i]
            inext = setpoints[i+1][0]
            x[icurr:inext, 0] = tcurr - band_width/2.
            x[icurr:inext, 1] = tcurr + band_width/2.
        self.comfort_band = pd.DataFrame(x)

    def __call__(
        self,
        batch: Batch,
        zone_temp: any,
        t: int,
        **kwargs
    ):

        bsz, _ = batch.comfort_min.shape

        # Cast comfort max to batch size of zone_temp
        comfort_max = (self.comfort_band.values[t, 1]
                       * torch.ones(bsz, zone_temp.shape[-1]))

        # We are always in minpower mode _unless_ too hot, then we cool.
        is_zone_above_comfort_band = zone_temp > comfort_max

        # Get indicator tensor for whether a sample is above the comfort band.
        _z = torch.zeros_like(zone_temp)
        mean_above_comfort_band = torch.max(
            _z, zone_temp - comfort_max).mean(axis=-1)
        is_mean_above_comfort_band = mean_above_comfort_band > 0.

        # Use minpower unless we're above the comfort band, then we cool.
        # Zone-level control (flow rate)
        action = to_torch(self.action_minpower, batch_size=bsz)
        batched_action = to_torch(self.action_cooling, batch_size=bsz)
        masked_action = batched_action[:, :-1][is_zone_above_comfort_band]
        action[:, :-1][is_zone_above_comfort_band] = masked_action
        # Discharge temp
        chiller_bool = is_mean_above_comfort_band
        action[:, -1][chiller_bool] = batched_action[:, -1][chiller_bool]

        return action, {}
