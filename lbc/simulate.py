import logging
import time

import numpy as np
import tqdm

import torch
import torch.nn.functional as F

from lbc.costs import stage_cost
import lbc.dynamics as dyn
from lbc.policies import Policy
from lbc.scenario import Scenario
from lbc.rollout import Rollout
from lbc.utils import to_torch


logger = logging.getLogger(__name__)


def simulate(
    policy: Policy,
    scenario: Scenario,
    batch_size: int,
    training: bool = True,
    use_tqdm: bool = False,
    pbar: any = None,
    pbar_epoch: any = None,
    pbar_loss: any = None,
    device: str = "cpu",
    **policy_kwargs
):
    """Runs the simulation using the given policy instance.

    Args:
        policy:  initialized policy instance
        scenario:  scenario instance
        batch_size:  batch size
        training:  whether we are training or simply running
        use_tqdm:  turn on/off use of tqdm
        pbar:  tqdm progress bar instance in case we want to update it
        pbar_epoch:  training epoch for progress bar
        pbar_loss: training loss for progress bar
        device:  torch device

    Returns:
        total_loss:  torch tensor for (batched) total loss
        rollout:  rollout instance containing all data from rollout
        policy_meta:  metadata dict returned by call to policy
    """

    # Keep track of wall time.
    tic = time.time()

    # Convenience variable.
    zone_model = scenario.zone_model

    # Make the batch of scenario data.
    batch = scenario.make_batch(
        batch_size, as_tensor=True, training=training, shuffle=True)

    # Useful dimensions.
    bsz, num_time, num_zone = batch.q_solar.shape

    # Set the initial conditions
    zone_temp = batch.zone_temp.to(device)   # (batch_size, num_zones)
    temp_oa = batch.temp_oa.to(device)       # (batch_size, 1)
    q_solar = batch.q_solar.to(device)       # (batch_size, 1)
    action_init = torch.tensor(
        scenario.action_init, dtype=torch.float32).to(device)
    action_init = action_init * torch.ones(
        bsz, *action_init.shape, dtype=torch.float32).to(device)
    x = torch.zeros(batch_size, num_zone).to(device)

    # Reshape the action and clip it to the min/max.
    action_min = to_torch(scenario.action_min, batch_size=bsz).to(device)
    action_max = to_torch(scenario.action_max, batch_size=bsz).to(device)

    # Initialize the total loss for each scenario (shape = (batch_size,))
    total_loss = torch.zeros(batch_size).to(device)

    # Rollout instance to log rollout data for analysis.
    rollout = Rollout(time_index=scenario.time_index)

    # Main loop over episode time steps.
    tt = range(num_time)
    tt = tqdm(tt) if use_tqdm else tt
    for t in tt:

        # Need to perform the filter update to "warm start" the model.
        if t == 0:

            # Initial feature vector.
            u = dyn.build_u_vector(
                model=zone_model,
                zone_temp=zone_temp,
                temp_oa=temp_oa[:, t].unsqueeze(-1),
                q_solar=q_solar[:, t, :],
                action=action_init,
                device=device
            )

            # We take two steps of filter update to get the initial state.
            for _ in range(2):
                x = dyn.filter_update(
                    x=x, u=u, zone_temp=zone_temp, model=zone_model,
                    device=device)

        # Forward pass.  Note that every Policy instance will get these values.
        # What each class does with them is up to the user.
        action, policy_meta = policy(
            scenario=scenario, batch=batch, t=t, u=u,
            x=x, zone_temp=zone_temp,  action_init=action_init,
            training=training, **policy_kwargs
        )

        # Reshape action to match batch shape.
        action = action.reshape(*action_min.shape).to(device)

        # Save the next initial action as this action.
        action_init = action.detach().clone()

        # Here we want to clip the action in a differentiable manner.
        # (We cannot use torch.max for this!)
        # Compute max(action, action_min) to clip from below.
        clipped_action = F.relu(action - action_min) + action_min
        # Compute min(action, action_max) to clip from above.
        clipped_action = -F.relu(-clipped_action + action_max) + action_max

        # Get current values of exogenous data and energy price.
        (temp_oa, q_solar, comfort_min, comfort_max, energy_price,
         predicted_energy_price) = batch.get_time(t, device=device)

        # Set power constraint limit and penalty depending on whether we're
        # using that demand response program.
        dr_program = scenario.dr_program
        pc_limit, pc_penalty = None, None
        if dr_program.program_type == 'PC':
            pc_limit = dr_program.power_limit.values[t][0] * \
                torch.ones(1).to(device)
            pc_penalty = dr_program.pc_penalty

        # Compute and accumulate the loss. If we are not training, don't
        # compute penalties for action soft constraint, we simply apply the
        # clipped actions as the control.
        cost_data = stage_cost(
            zone_model=zone_model,
            temp_oa=temp_oa,
            zone_temp=zone_temp,
            comfort_min=comfort_min,
            comfort_max=comfort_max,
            energy_price=energy_price,
            action=action,
            clipped_action=clipped_action,
            action_penalty=scenario.action_penalty if training else 0.,
            action_min=action_min,
            action_max=action_max,
            comfort_penalty=scenario.comfort_penalty,
            pc_penalty=pc_penalty,
            pc_limit=pc_limit,
            device=device)

        # Evolve the "true" dynamics given the (clipped) action, current
        # exogenous data, and zone models.
        x, zone_temp, _ = dyn.dynamics(
            x=x, zone_temp=zone_temp, action=clipped_action, temp_oa=temp_oa,
            q_solar=q_solar, model=zone_model, device=device)

        # Compute any secondary metrics
        comfort_viol_deg_hr = ((cost_data.comfort_viol_upper.sum(axis=-1)
                                + cost_data.comfort_viol_lower.sum(axis=-1))
                               * scenario.hours_per_timestep)

        # Update the rollout instance with any/all data we might want to look
        # at later. TODO:  How to make less ugly?
        # Keeps track of batched variables.
        rollout.update_batched(
            temp_oa=temp_oa,
            zone_temp=zone_temp,
            comfort_min=comfort_min,
            comfort_max=comfort_max,
            energy_price=energy_price,
            predicted_energy_price=predicted_energy_price,
            action=action,
            clipped_action=clipped_action,
            action_min=action_min,
            action_max=action_max,
            comfort_viol_deg_hr=comfort_viol_deg_hr,
            **cost_data.__dict__
        )

        # Keeps track of scalar variables.
        rollout.update_scalar(
            action_penalty=scenario.action_penalty if training else 0.,
            comfort_penalty=scenario.comfort_penalty,
            pc_penalty=pc_penalty,
            pc_limit=pc_limit)

        # Average loss per sample
        total_loss += cost_data.total_cost

        # Optionally update the progress bar from calling scope.
        if pbar is not None:
            _loss = total_loss.mean().item()
            pbar_loss = pbar_loss if pbar_loss is not None else np.inf
            pbar.set_description(
                f"epoch={pbar_epoch}|loss={pbar_loss:1.3e}"
                + f"|step={t+1}/{num_time}|{_loss:1.3e}")

    # Average loss per time step
    total_loss /= num_time

    # Stack the arrays in the rollout data
    rollout.finalize()

    # Add cpu time to policy metadata
    policy_meta = {"cpu_time": time.time() - tic}

    return total_loss, rollout, policy_meta
