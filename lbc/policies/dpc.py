import logging
from math import ceil, floor
from typing import Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from lbc.policies import Policy
from lbc.scenario import Scenario, Batch
from lbc.utils import to_torch


logger = logging.getLogger(__file__)


class SingleStepDPCModelRNN(nn.Module):

    num_zones: int = 5

    def __init__(
        self,
        num_time_windows: int,
        num_episode_steps: int,
        hidden_dim: int = 64,
        embed_dim: int = 128,
        price_forecast_dim: int = None,
        **kwargs
    ):

        super().__init__()

        self.num_time_windows = num_time_windows
        self.num_episode_steps = num_episode_steps
        self.steps_per_window = ceil(num_episode_steps / num_time_windows)
        if price_forecast_dim is not None:
            self.price_forecast_dim = price_forecast_dim
        else:
            self.price_forecast_dim = 1
        self.hidden_dim = hidden_dim
        self.output_dim = self.num_zones + 1

        # Input size, needs to be updated if we are looking ahead on prices
        self.input_size = 2 * self.num_zones + 4 * \
            self.num_zones + 3 + self.price_forecast_dim
        # self.input_size = self.num_zones + 3 + self.price_forecast_dim

        self.embed_dim = embed_dim
        if self.embed_dim > 1:
            self.embed = nn.Embedding(
                num_time_windows, embedding_dim=embed_dim)
        else:
            self.embed = None

        if self.embed is None:
            self.first_dim = 2 + self.input_size
        else:
            self.first_dim = embed_dim + self.input_size

        self.rnn = nn.LSTM(
            self.first_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, *, x, u, t, last_energy_price, zone_temp, temp_oa,
                predicted_energy_price=None, pc_limit=None):

        bsz = x.shape[0]

        if self.embed is not None:
            x_t = self.embed(t)
        else:
            x_t_sin = torch.sin(
                2 * np.pi / self.num_episode_steps * t.unsqueeze(1))
            x_t_cos = torch.cos(
                2 * np.pi / self.num_episode_steps * t.unsqueeze(1))
            x_t = torch.cat((x_t_sin, x_t_cos), axis=-1)

        if pc_limit is None:
            pc_limit = torch.zeros_like(last_energy_price)

        if predicted_energy_price is None:
            predicted_energy_price = torch.zeros_like(
                bsz, self.price_forecast_dim)

        x_pi = torch.cat(
            (x, zone_temp, u.reshape(-1, u.shape[1] * u.shape[2]),
             last_energy_price, temp_oa, predicted_energy_price, pc_limit),
            axis=-1)

        if t[0] == 0:
            self.hn = torch.randn(1, bsz, self.hidden_dim)
            self.cn = torch.randn(1, bsz, self.hidden_dim)

        logits = torch.cat((x_t, x_pi), axis=-1).unsqueeze(1)
        logits, (self.hn, self.cn) = self.rnn(logits, (self.hn, self.cn))
        logits = nn.Flatten(start_dim=1, end_dim=-1)(logits)
        logits = torch.sigmoid(self.linear2(
            logits)).reshape(bsz, self.output_dim)

        return logits


class SingleStepDPCModel(nn.Module):

    num_zones: int = 5

    def __init__(
        self,
        num_time_windows: int,
        num_episode_steps: int,
        hidden_dim: int = 64,
        embed_dim: int = 128,
        price_forecast_dim: int = None,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.num_time_windows = num_time_windows
        self.num_episode_steps = num_episode_steps
        self.steps_per_window = ceil(num_episode_steps / num_time_windows)
        if price_forecast_dim is not None:
            self.price_forecast_dim = price_forecast_dim
        else:
            self.price_forecast_dim = 1

        # Input size, needs to be updated if we are looking ahead on prices
        # TODO: Add explanation for the dimensions below
        self.input_size = (2 * self.num_zones + 4 * self.num_zones
                           + 3 + self.price_forecast_dim)

        # # Create the policy layers.
        self.embed = None
        if embed_dim is not None:
            self.embed = nn.Embedding(num_time_windows,
                                      embedding_dim=embed_dim)
            self.first_dim = embed_dim + self.input_size
        else:
            self.embed = None
            self.first_dim = self.input_size + 1

        self.policy1 = nn.Linear(self.first_dim, hidden_dim)
        self.policy2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy3 = nn.Linear(hidden_dim, self.num_zones + 1)

    def forward(self, *, x, u, t, last_energy_price, zone_temp, temp_oa,
                predicted_energy_price=None, pc_limit=None):

        bsz = x.shape[0]

        if self.embed is not None:
            x_t = self.embed(t)
        else:
            x_t = t[0] * torch.ones(bsz, 1)

        if pc_limit is None:
            pc_limit = torch.zeros_like(last_energy_price)

        if predicted_energy_price is None:
            predicted_energy_price = torch.zeros_like(
                bsz, self.price_forecast_dim)

        x_pi = torch.cat(
            (x, zone_temp, u.reshape(-1, u.shape[1] * u.shape[2]),
             last_energy_price, temp_oa, predicted_energy_price, pc_limit),
            axis=-1)

        logits = torch.cat((x_t, x_pi), axis=-1)
        logits = F.relu(self.policy1(logits))
        logits = F.relu(self.policy2(logits))
        logits = torch.sigmoid(self.policy3(logits))

        return logits


class DPCPolicy(Policy):

    def __init__(
        self,
        model_config: dict,
        scenario: Scenario,
        device: str = "cpu",
        exploration_noise_std: float = None,
        **kwargs
    ):

        super().__init__()

        self.device = device

        # Update the model config with num episode steps to compute time embed.
        model_config.update({
            "num_episode_steps": scenario.num_episode_steps
        })

        if ("model_type" not in model_config
                or model_config["model_type"] == "fully_connected"):
            self.model = SingleStepDPCModel(**model_config).to(device)
        elif model_config["model_type"] == "rnn":
            self.model = SingleStepDPCModelRNN(**model_config).to(device)
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

        t_la = [(t + s) % num_time
                for s in range(self.model.price_forecast_dim)]
        pep = batch.predicted_energy_price[:, t_la]
        predicted_energy_price = pep.reshape((-1, len(t_la))).to(device)

        # we assume price changes every one hour, so minus 12 means last hour.
        t_last = (t - 12) % num_time
        lep = batch.energy_price[:, t_last]
        last_energy_price = lep.reshape((-1, 1)).to(device)

        t_torch = floor(t / self.model.steps_per_window) \
            * torch.ones(bsz, dtype=torch.long).to(device)  # / num_time

        temp_oa = batch.temp_oa[:, t].reshape(-1, 1).to(device)

        if scenario.dr_program.program_type == "PC":
            pc_limit = (scenario.dr_program.power_limit.values[t][0]
                        * torch.ones(bsz, 1))
        else:
            pc_limit = torch.zeros_like(last_energy_price)

        logits = self.model(
            x=x, u=u, t=t_torch, last_energy_price=last_energy_price,
            predicted_energy_price=predicted_energy_price,
            zone_temp=zone_temp, temp_oa=temp_oa, pc_limit=pc_limit)

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
