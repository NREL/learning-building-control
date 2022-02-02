import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from lbc.policies import Policy
from lbc.scenario import Scenario, Batch
from lbc.utils import to_torch


logger = logging.getLogger(__file__)


class SingleStepDPCModel(nn.Module):

    num_zones: int = 5

    def __init__(
        self,
        num_intervals: int,
        hidden_dim: int = 64,
        embed_dim: int = 16,
        **kwargs
    ):

        super().__init__(**kwargs)

        input_size = 2 * self.num_zones + 4 * self.num_zones + 1
        self.num_intervals = num_intervals

        # # Create the policy layers.
        self.embed = nn.Embedding(num_intervals, embedding_dim=embed_dim)
        self.embed_layer = nn.Linear(embed_dim, hidden_dim)
        self.policy1 = nn.Linear(input_size, hidden_dim)
        self.policy2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.policy3 = nn.Linear(hidden_dim, self.num_zones + 1)

    def forward(self, *, x, u, t, energy_price, zone_temp):

        t_embed = self.embed(t)
        x_t = self.embed_layer(t_embed)

        x_pi = torch.cat(
            (x, zone_temp, u.reshape(-1, u.shape[1] * u.shape[2]),
             energy_price),
            axis=-1)
        x_pi = F.relu(self.policy1(x_pi))

        action = torch.cat((x_t, x_pi), axis=-1)
        action = F.relu(self.policy2(action))
        action = self.policy3(action)

        return action


class DPCPolicy(Policy):

    def __init__(
        self,
        model_config: dict,
        device: str = "cpu",
        exploration_noise_std: float = None,
        **kwargs
    ):

        super().__init__()

        self.device = device
        self.model = SingleStepDPCModel(**model_config).to(device)
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

        energy_price = to_torch(
            batch.predicted_energy_price[:, t].reshape((-1, 1))).to(device)

        t_torch = (t // self.model.num_intervals) \
            * torch.ones(bsz, dtype=torch.long).to(device)  # / num_time

        action = self.model(x=x, u=u, t=t_torch,
                            energy_price=energy_price, zone_temp=zone_temp)

        if self.exploration_noise_std is not None and training:
            noise = self.exploration_noise_std \
                * torch.randn(*action.shape).to(self.device)
            action += noise

        return action, {}
