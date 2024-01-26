import logging
from typing import Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from lbc.building_env import STEPS_PER_HOUR
from lbc.building_env import ZONE_TEMP_BOUNDS
from lbc.building_env import POWER_LIMIT_BOUNDS
from lbc.building_env import PRICE_BOUNDS, Q_SOLAR_BOUNDS, OUTDOOR_TEMP_BOUNDS
from lbc.policies import Policy
from lbc.scenario import Scenario, Batch
from lbc.utils import to_torch


logger = logging.getLogger(__file__)


class SingleStepDPCModel(nn.Module):

    num_zones: int = 5

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        **kwargs
    ):

        super().__init__(**kwargs)

        # # Create the policy layers.
        self.policy1 = nn.Linear(input_dim, hidden_dim)
        self.policy2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy3 = nn.Linear(hidden_dim, self.num_zones + 1)

    def forward(self, obs_input):

        logits = F.relu(self.policy1(obs_input))
        logits = F.relu(self.policy2(logits))
        logits = torch.sigmoid(self.policy3(logits))

        return logits


class DPCPolicy(Policy):

    def __init__(
        self,
        model_config: dict,
        scenario: Scenario,
        num_lookahead_steps: int = 24,
        device: str = "cpu",
        exploration_noise_std: float = None,
        **kwargs
    ):

        super().__init__()

        self.device = device

        # Update the model config with num episode steps to compute time embed.
        self.num_lookahead_steps = num_lookahead_steps
        dr_type = scenario.dr_program.program_type
        if dr_type == 'TOU':
            dim_obs = 5 + 5 + 6 * self.num_lookahead_steps + 26
        elif dr_type == 'PC':
            dim_obs = 5 + 5 + 7 * self.num_lookahead_steps + 26
        elif dr_type == 'RTP':
            dim_obs = 5 + 5 + 7 * self.num_lookahead_steps + 26

        model_config.update({
            "input_dim": dim_obs
        })

        if ("model_type" not in model_config
                or model_config["model_type"] == "fully_connected"):
            self.model = SingleStepDPCModel(**model_config).to(device)
        else:
            raise NotImplementedError("Model type not supported")

        self.exploration_noise_std = exploration_noise_std

    def __call__(
        self,
        scenario: Scenario,
        batch: Batch,
        t: int,
        x: any,
        u: any,
        zone_temp: any,
        training: bool = True,
        **kwargs
    ) -> Tuple[torch.tensor, dict]:

        bsz, num_time, _ = batch.q_solar.shape

        device = self.device

        def normalize(tensor, bounds):
            mean = np.mean(bounds)
            val_range = bounds[1] - mean
            normalized_tensor = (tensor - mean) / val_range
            return normalized_tensor

        normalized_zone_temp = normalize(
            zone_temp, ZONE_TEMP_BOUNDS).to(device)

        temp_oa = batch.temp_oa[:, t: t + self.num_lookahead_steps]
        need_padding = self.num_lookahead_steps - temp_oa.shape[1]
        temp_oa = torch.cat(
            [temp_oa] + [temp_oa[:, -1].reshape(-1, 1)] * need_padding,
            axis=-1)
        normalized_temp_oa = normalize(temp_oa, OUTDOOR_TEMP_BOUNDS).to(device)

        q_solar = batch.q_solar[:, t: t + self.num_lookahead_steps, :]
        need_padding = self.num_lookahead_steps - q_solar.shape[1]
        q_solar = torch.nn.functional.pad(q_solar, (0, 0, 0, need_padding),
                                          'replicate')
        q_solar_flatten = torch.flatten(q_solar, start_dim=1)
        normalized_q_solar = normalize(q_solar_flatten,
                                       Q_SOLAR_BOUNDS).to(device)

        one_hot_t = [0] * 24
        one_hot_t[int(t / STEPS_PER_HOUR)] = 1
        degree = t / 288.0 * 2 * np.pi
        trignometric_time = [np.sin(degree), np.cos(degree)]

        time_encoding = torch.Tensor(
            [one_hot_t + trignometric_time for _ in range(bsz)]
        ).to(device)

        # Assemble observation
        batched_observation = torch.cat(
            (x, normalized_zone_temp, normalized_temp_oa, normalized_q_solar,
             time_encoding), axis=-1
        ).to(device)

        if scenario.dr_program.program_type == "PC":
            pc_limit = scenario.dr_program.power_limit.values[
                t: t + self.num_lookahead_steps]
            batch_pc_limit = torch.Tensor(
                [pc_limit.reshape((np.dot(*pc_limit.shape),))
                 for _ in range(bsz)])
            need_padding = self.num_lookahead_steps - batch_pc_limit.shape[1]
            batch_pc_limit = torch.nn.functional.pad(
                batch_pc_limit, (0, need_padding), 'replicate')
            batch_pc_limit = normalize(batch_pc_limit, POWER_LIMIT_BOUNDS)

            batched_observation = torch.cat(
                (batched_observation, batch_pc_limit), axis=-1
            )
        elif scenario.dr_program.program_type == "RTP":
            price_forecast = batch.predicted_energy_price[
                :, t: t + self.num_lookahead_steps]
            need_padding = self.num_lookahead_steps - price_forecast.shape[1]
            price_forecast = torch.nn.functional.pad(
                price_forecast, (0, need_padding), 'replicate')
            normalized_price_forecast = normalize(price_forecast, PRICE_BOUNDS)
            batched_observation = torch.cat(
                (batched_observation, normalized_price_forecast), axis=-1
            )

        logits = self.model(batched_observation)

        # squash to range
        amin = to_torch(scenario.action_min, batch_size=bsz)
        amax = to_torch(scenario.action_max, batch_size=bsz)
        action = amin + logits * (amax - amin)

        # Optionally add exploration noise during training
        if self.exploration_noise_std is not None and training:
            noise = self.exploration_noise_std \
                * torch.randn(*action.shape).to(self.device)
            action += noise

        return action, {}
