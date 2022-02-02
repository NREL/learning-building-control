from typing import Tuple

import torch
from torch.nn.parameter import Parameter

from lbc.utils import to_torch


def build_u_vector(
    model: dict,
    zone_temp: torch.tensor,
    temp_oa: torch.tensor,
    q_solar: torch.tensor,
    action: torch.tensor = None,
    q_cool: torch.tensor = None,
    device: str = "cpu"
) -> torch.tensor:
    """Returns the u-vector based on temps, actions, and exogenous data.
    If action is None, q_cool must be provided.

    Args:
        models: dict of zone model parameters
        zone_temp:  zone temperatures, shape = [batch_size, num_zones]
        action: control actions, shape = [batch_size, num_zones + 1]
        temp_oa: outdoor air temperature, shape = [batch_size, 1]
        q_solar: solar irradiance, shape = [batch_size, num_zones]
        q_cool:  cooling energy (if action not supplied),
            shape = [batch_size, num_zones]

    Returns:
        u: u feature vector, shape = [batch_size, num_zones, num_features]

    The feature set for u is:
        0: difference between outdoor temp and zone temp
        1: q_solar for each zone
        2: difference between zone temp and selected neighbor
        3: cooling energy
    """

    bsz, Z = zone_temp.shape

    # Create the u_ vector that contains the features for each zone.
    # TODO: vectorize!
    u = torch.zeros(bsz, Z, 4).to(device)
    for z in range(Z):
        u[:, z, 0] = temp_oa[:, 0] - zone_temp[:, z]
        u[:, z, 3] = q_solar[:, z]
        u[:, z, 2] = zone_temp[:, model["u_neighbor"][z]] - zone_temp[:, z]

        if action is None:
            u[:, z, 1] = q_cool[:, z]
        else:
            u[:, z, 1] = action[:, z] * (action[:, -1] - zone_temp[:, z])

    return u


def state_update(
    x: torch.tensor,
    u: torch.tensor,
    model: dict = None,
    A: Parameter = None,
    B: Parameter = None,
    device: str = "cpu"
) -> torch.tensor:
    """Returns the next state based on current state and u-vector.

    Args:
        x: state vector, shape = [batch_size, num_zones]
        u: u-feature vector, shape = [batch_size, num_zones, num_features]
        model: dict of zone model parameters
        A,B: model Parameters

    Returns:
        x: Ax + Bu, shape = [batch_size, num_zones]
    """

    A = A if A is not None else to_torch(model["A"])
    B = B if B is not None else to_torch(model["B"])

    A = A.to(device)
    B = B.to(device)

    # print(A.shape, B.shape, B.T.shape, x.shape, u.shape)
    # print((A * x).shape, (B.T * u).shape, (B.T * u).sum(axis=-1).shape)

    return (A * x) + (B.T * u).sum(axis=-1)


def filter_update(
    x: torch.tensor,
    u: torch.tensor,
    zone_temp: torch.tensor,
    model: dict = None,
    A: Parameter = None,
    B: Parameter = None,
    C: Parameter = None,
    K: Parameter = None,
    mean_output: Parameter = None,
    device: str = "cpu"
) -> torch.tensor:
    """Perform the filter update given current state, temps, and u-vector.

    Args
        x: state vector, shape = [batch_size, num_zones]
        u: u: u-feature vector, shape = [batch_size, num_zones]
        zone_temp:  zone temperatures for each size,
            shape = [batch_size, num_zones]
        model: dict of zone model parameters
        A,B,C,K,mean_output:  model Parameters (optional)

    Returns:
        x: updated state vector, shape = [batch_size, num_zones]
    """

    C = C if C is not None else to_torch(model["C"])
    K = K if K is not None else to_torch(model["K"])
    mean_output = (mean_output
                   if mean_output is not None
                   else to_torch(model["mean_output"]))

    C = C.to(device)
    K = K.to(device)
    mean_output = mean_output.to(device)

    x = state_update(x, u, model=model, A=A, B=B, device=device)

    yhat_kplus1 = C * x

    y_actual = zone_temp - mean_output

    x += K * (y_actual - yhat_kplus1)

    return x


def temp_dynamics(
    x: torch.tensor,
    model: dict = None,
    C: Parameter = None,
    mean_output: Parameter = None,
    device: str = "cpu"
) -> torch.tensor:
    """Returns the new zone temperatures based on the current state.

    Args:
        x: state vector, shape = [batch_size, num_zones]
        model: dict of zone model parameters
        C,mean_output: model Paramters (optional)

    Returns:
        zone_temp: zone temperatures, shape = [batch_size, num_zones]
    """

    C = C if C is not None else to_torch(model["C"])
    mean_output = (mean_output
                   if mean_output is not None
                   else to_torch(model["mean_output"]))

    C = C.to(device)
    mean_output = mean_output.to(device)

    return C * x + mean_output


def dynamics(
    x: torch.tensor = None,
    zone_temp: torch.tensor = None,
    action: torch.tensor = None,
    temp_oa: torch.tensor = None,
    q_solar: torch.tensor = None,
    q_cool: torch.tensor = None,
    model: dict = None,
    A: Parameter = None,
    B: Parameter = None,
    C: Parameter = None,
    mean_output: Parameter = None,
    device: str = "cpu"
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Returns the new state and zone temps based on current state, temps,
    actions and exogenous data.

    Args:
        x: state vector, shape = [batch_size, num_zones]
        zone_temp:  zone temperatures for each size,
            shape = [batch_size, num_zones]
        action: control actions, shape = [batch_size, num_zones + 1]
        temp_oa: outdoor air temperature, shape = [batch_size, 1]
        q_solar: solar irradiance, shape = [batch_size, num_zones]
        q_cool:  cooling energy (if action not supplied),
            shape = [batch_size, num_zones]
        model: dict of zone model parameters
        A,B,C,mean_output:  model Parameters (optional)

    Returns:
        x_out: updated state, shape = [batch_size, num_zones]
        zone_temp: zone temperatures, shape = [batch_size, num_zones]
        u: feature vector, shape = [batch_size, num_zones]
    """

    u = build_u_vector(
        model=model, zone_temp=zone_temp, temp_oa=temp_oa,
        action=action, q_solar=q_solar, q_cool=q_cool, device=device)
    x = state_update(model=model, x=x, u=u, A=A, B=B, device=device)
    zone_temp = temp_dynamics(model=model, x=x, C=C, mean_output=mean_output,
                              device=device)

    return x, zone_temp, u
