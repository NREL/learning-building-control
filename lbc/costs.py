from dataclasses import dataclass
import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class CostData:
    total_cost: torch.tensor
    comfort_viol_lower: torch.tensor
    comfort_viol_upper: torch.tensor
    action_viol_lower: torch.tensor
    action_viol_upper: torch.tensor
    action_viol_cost: torch.tensor
    chiller_power: torch.tensor
    fan_power: torch.tensor
    total_power: torch.tensor
    power_cost: torch.tensor
    comfort_cost: torch.tensor
    action_viol_cost: torch.tensor
    pc_violation_cost: torch.tensor


def stage_cost(
    zone_model: dict,
    action: torch.tensor,
    clipped_action: torch.tensor,  # we assume a clipped action is passed
    temp_oa: torch.tensor,
    zone_temp: torch.tensor,
    energy_price: torch.tensor,
    action_penalty: torch.tensor,
    action_min: torch.tensor,
    action_max: torch.tensor,
    comfort_penalty: torch.tensor,
    comfort_min: float,
    comfort_max: float,
    pc_penalty: float,  # TODO: @xzhang2, should this be float or tensor?
    pc_limit: float,
    device: str = "cpu",
) -> torch.tensor:

    hvac_cop, fan_coeff_1, fan_coeff_2 = zone_model['hvac_parameters']
    delta_t = zone_model['delta_t']

    # Fan power.  We need to clip action
    fan_power = (fan_coeff_1 * (clipped_action[:, :-1].sum(axis=-1))**3
                 + fan_coeff_2)

    # Chiller power
    chiller_power = (hvac_cop * clipped_action[:, :-1].sum(axis=-1)
                     * F.relu(temp_oa.squeeze() - clipped_action[:, -1]))

    # Total power and power cost
    total_power = fan_power + chiller_power
    power_cost = energy_price.squeeze() * total_power * delta_t

    # Comfort bound violations and cost
    comfort_viol_lower = F.relu(comfort_min - zone_temp)
    comfort_viol_upper = F.relu(zone_temp - comfort_max)
    comfort_cost = comfort_penalty * \
        torch.sum(comfort_viol_lower.pow(2) +
                  comfort_viol_upper.pow(2), axis=-1)

    # Action bound violations and cost
    action_viol_lower = F.relu(action_min - action)
    action_viol_upper = F.relu(action - action_max)
    action_viol_cost = action_penalty * \
        torch.sum(action_viol_lower.pow(2) + action_viol_upper.pow(2), axis=-1)

    total_cost = power_cost + comfort_cost + action_viol_cost

    # DR power constraint violation
    if pc_limit is not None:
        pc_violation = F.relu(total_power - pc_limit)
        pc_violation_cost = pc_penalty * pc_violation.pow(2)
    else:
        pc_violation_cost = torch.zeros_like(total_cost)
    total_cost += pc_violation_cost

    return CostData(
        total_cost, comfort_viol_lower, comfort_viol_upper,
        action_viol_lower, action_viol_upper, action_viol_cost,
        chiller_power, fan_power, total_power, power_cost,
        comfort_cost, pc_violation_cost)
