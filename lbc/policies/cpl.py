import logging

from typing import Tuple

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

from lbc.batch import Batch
from lbc.demand_response import DemandResponseProgram
from lbc.policies import Policy
from lbc.scenario import Scenario
from lbc.utils import to_torch


logger = logging.getLogger(__name__)


class CPLModel:

    def __init__(
        self,
        scenario: Scenario = None,
        dr_program: DemandResponseProgram = None,
        step_idx: int = None,
        lookahead: int = None,
        wrap_horizon: bool = True,
        use_value_function: bool = True,
        value_scale: float = 1.0,
        **kwargs
    ):

        self.zone_model = scenario.zone_model
        self.scenario = scenario
        self.dr_program = dr_program
        self.step_idx = step_idx
        self.lookahead = lookahead
        self.wrap_horizon = wrap_horizon
        self.use_value_function = use_value_function
        self.value_scale = value_scale

        # Get the action space bounds for use in action constraints
        self.action_min = scenario.action_min
        self.action_max = scenario.action_max

        self.problem = self.create_problem()
        self.problem.to("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_problem(self):

        # Index dimensions
        T = self.lookahead
        Z = len(self.zone_model["A"])
        F = self.zone_model["B"].shape[0]

        # Conveninece variables
        A = self.zone_model["A"]
        B = self.zone_model["B"]
        C = self.zone_model["C"].reshape(1, -1)
        mean_output = self.zone_model["mean_output"].reshape(1, -1)
        nbrs = self.zone_model["u_neighbor"]
        comfort_penalty = self.scenario.comfort_penalty

        # VARIABLES
        # State variables
        zone_temp = cp.Variable((T, Z), nonneg=True, name="zone_temp")
        x = cp.Variable((T, Z), name="x")
        u0 = cp.Variable((T, Z), name="u0")
        u1 = cp.Variable((T, Z), name="u1")
        u2 = cp.Variable((T, Z), name="u2")
        u3 = cp.Variable((T, Z), name="u3")

        # Action variables
        action = cp.Variable((T, Z+1), nonneg=True, name="action")
        mpc_action = cp.Variable((1, Z+1), nonneg=True, name="mpc_action")
        ms_dot_sum = cp.Variable(T, nonneg=True, name="ms_dot_sum")
        discharge_temp = cp.Variable(T, nonneg=True, name="discharge_temp")

        # Dummy variables to enable DPP.
        zone_temp_init_v = cp.Variable((1, Z), nonneg=True,
                                       name="zone_temp_init_v")  # to make dpp
        action_init_v = cp.Variable((1, Z+1), nonneg=True,
                                    name="action_init_v")  # to make dpp

        # Slack variables for soft constraint on comfort.
        comfort_viol_lower = cp.Variable((T, Z), nonneg=True,
                                         name="comfort_viol_lower")
        comfort_viol_upper = cp.Variable((T, Z), nonneg=True,
                                         name="comfort_viol_upper")

        # PARAMETERS
        # Terminal penalty
        if self.use_value_function:
            q = cp.Parameter(Z, name="q")
            Q_sqrt = cp.Parameter((Z, Z), name="Q_sqrt", PSD=True)

        # Exogenous data.  When env data is used, the index is wrapped such
        # that data is always available in the lookahead.  When in doubt,
        # create a parameter with 2 dimensions to enable broadcasting.
        zone_temp_init = cp.Parameter((1, Z), nonneg=True,
                                      name="zone_temp_init")
        x_init = cp.Parameter((1, Z), name="x_init")
        temp_oa = cp.Parameter((T, 1), nonneg=True, name="temp_oa")
        q_solar = cp.Parameter((T, Z), nonneg=True, name="q_solar")
        action_init = cp.Parameter((1, Z+1), nonneg=True, name="action_init")
        tou = cp.Parameter((T, 1), nonneg=True, name="tou")
        power_limit = cp.Parameter((T, 1), nonneg=True, name="power_limit")
        action_init_tou = cp.Parameter((1, Z+1), nonneg=True,
                                       name="action_init_tou")
        temp_oa_tou = cp.Parameter((T, 1), nonneg=True, name="temp_oa_tou")
        comfort_min = cp.Parameter((T, 1), nonneg=True, name="comfort_min")
        comfort_max = cp.Parameter((T, 1), nonneg=True, name="comfort_max")

        # CONSTRAINTS
        # Bound constraints
        constraints = []

        # MPC action is first time step
        constraints += [mpc_action[0, :] == action[0, :]]

        # Control bounds
        constraints += [action >= self.scenario.action_min.reshape(1, -1)]
        constraints += [action <= self.scenario.action_max.reshape(1, -1)]

        # Zone temp comfort violations
        constraints += [zone_temp + comfort_viol_lower >= comfort_min]
        constraints += [zone_temp - comfort_viol_upper <= comfort_max]

        # Initial conditions
        constraints += [zone_temp[0, :] == zone_temp_init[0, :]]
        constraints += [x[0, :] == x_init[0, :]]

        # Create a dummy variable to allow DPP formulation.
        constraints += [zone_temp_init_v == zone_temp_init]
        constraints += [action_init_v == action_init]

        # Total fan power per time step.
        # original version not dcp so moved cube into objective
        constraints += [ms_dot_sum == cp.sum(action[:, :-1], axis=1)]
        constraints += [discharge_temp == action[:, -1]]

        # Dimensions | zone_temp: [T-1, 5], C: [1, 5], x: [T-1, 5],
        # mean_output: [1, 5]
        # Use cp.multiply to broadcast pointwise multiplication across first
        # dimension.
        constraints += [zone_temp[1:, :] ==
                        cp.multiply(C, x[1:, :]) + mean_output]

        u = [u0, u1, u2, u3]
        for z in range(Z):
            constraints += [
                x[1:, z] ==
                A[z] * x[:-1, z] +
                cp.sum([B[i, z] * u[i][:-1, z] for i in range(F)])
            ]

        # Definitions. These constraints are derived from the `build_u_vec`
        # method in reduced_order_model.py
        constraints += [u0 == temp_oa - zone_temp]

        # This feature is the difference between current zone and the selected
        # neighbor zone.
        constraints += [u2 == zone_temp[:, nbrs] - zone_temp]

        # Q solar definition.
        constraints += [u3 == q_solar]

        # Q cooling constraints.  Using numpy broadcast / slicing voodoo to
        # align dimensions but avoid having to tile or create unnecessary
        # copies. u1 has dimension (time, zone).
        constraints += [
            u1 ==
            cp.multiply(action[:, :Z], action_init[:, -1] - zone_temp_init)
            + cp.multiply(-action_init[:, :Z], action_init_v[:, -1]
                          - zone_temp_init_v)
            + cp.multiply(action_init[:, :Z], action[:, Z:Z+1])
            - cp.multiply(action_init[:, :Z], zone_temp_init_v[:, :])
        ]

        # Fan power
        hvac_cop, fan_coeff_1, fan_coeff_2 = self.zone_model['hvac_parameters']
        fan_power_notDcp = fan_coeff_1 * \
            cp.power(cp.sum(action[:, :-1], axis=1), 3) + fan_coeff_2

        tou_weighted_chiller_power = \
            hvac_cop * (
                cp.multiply(cp.sum(action[:, :Z], axis=1), temp_oa_tou[:, 0])
                - cp.sum(
                    cp.multiply(action[:, :Z] - action_init_v[:, :Z],
                                action_init_tou[:, -1])
                    + cp.multiply(action_init_tou[:, :Z],  action[0:T, Z:Z+1]),
                    axis=1
                )
            )

        # Chiller power needs to be non-negative.
        constraints += [
            hvac_cop * (
                cp.multiply(cp.sum(action[:, :Z], axis=1), temp_oa[:, 0])
                - cp.sum(
                    cp.multiply(action[:, :Z] - action_init_v[:, :Z],
                                action_init[:, -1])
                    + cp.multiply(action_init[:, :Z],  action[0:T, Z:Z+1]),
                    axis=1
                )
            ) >= 0.0
        ]

        # OBJECTIVE
        # tou weighted chiller power
        delta_t = self.zone_model['delta_t']
        objective = cp.sum(tou_weighted_chiller_power) * delta_t
        objective += cp.sum(tou[:, 0] * fan_power_notDcp) * delta_t  # P_fan
        objective += comfort_penalty * \
            cp.sum_squares(
                comfort_viol_lower[1:, :] + comfort_viol_upper[1:, :])

        if self.dr_program.program_type == 'PC':
            assert power_limit is not None
            power_viol = cp.Variable((T, 1), nonneg=True, name="power_viol")
            chiller_power = hvac_cop * (
                cp.multiply(cp.sum(action[:, :Z], axis=1), temp_oa[:, 0])
                - cp.sum(
                    cp.multiply(action[:, :Z] - action_init_v[:, :Z],
                                action_init[:, -1])
                    + cp.multiply(action_init[:, :Z], action[0:T, Z:Z + 1]),
                    axis=1)
            )
            # Assume in PC program energy price/TOU is constant.
            constraints += [chiller_power
                            + fan_power_notDcp
                            - power_viol[:, 0]
                            <= power_limit[:, 0]]

            objective += self.dr_program.pc_penalty * \
                cp.sum_squares(power_viol)

        if self.use_value_function:
            objective += q @ zone_temp[-1, :] + \
                cp.sum_squares(Q_sqrt @ zone_temp[-1, :])

        problem = cp.Problem(cp.Minimize(objective), constraints)

        # Problem parameters
        parameters = [
            temp_oa, q_solar, zone_temp_init, action_init, x_init, tou,
            temp_oa_tou, action_init_tou, comfort_min, comfort_max]
        if self.dr_program.program_type == 'PC':
            parameters.append(power_limit)

        if self.use_value_function:
            parameters += [q, Q_sqrt]

        return CvxpyLayer(problem, variables=[mpc_action],
                          parameters=parameters)


class CPLPolicy(Policy):

    def __init__(
        self,
        scenario: Scenario = None,
        lookahead: int = None,
        wrap_horizon: bool = True,
        use_value_function: bool = True,
        value_scale: float = 1.0,
        # number of intervals for value function
        num_value_interval_steps: float = 12,
        solver_args: dict = {},
        device: str = "cpu",
        **kwargs
    ):

        super().__init__()

        self.scenario = scenario
        self.dr_program = scenario.dr_program
        self.wrap_horizon = wrap_horizon
        self.lookahead = lookahead
        self.use_value_function = use_value_function
        self.value_scale = value_scale
        self.num_value_interval_steps = num_value_interval_steps
        self.solver_args = solver_args

        self.cpl = CPLModel(
            scenario=scenario,
            dr_program=self.dr_program,
            lookahead=lookahead,
            wrap_horizon=wrap_horizon,
            use_value_function=use_value_function,
            value_scale=value_scale)

        self.cpl.problem.to(device)

    def __call__(
        self,
        batch: Batch,           # current batch of scenario data
        q: torch.tensor,        # q-vector
        Q_sqrt: torch.tensor,   # Q_sqrt tensor
        t: int,                 # current time index
        x: any,                 # current state
        zone_temp: any,         # current zone temperatures
        action_init: any,       # initial action
        device: str = "cpu",
        training: bool = True,
        **kwargs
    ) -> Tuple[torch.tensor, dict]:

        # Handle requested wrapping strategy
        total_rows = self.scenario.num_episode_steps
        if self.wrap_horizon:
            # If wrapping, use the full lookahead and wrap indices into
            # exogenous data
            wrapped_index = [x %
                             total_rows for x in range(t, t + self.lookahead)]
        else:
            # Otherwise, shorten the lookahead to end with the data
            T = min(self.lookahead, total_rows - t)
            wrapped_index = list(range(t, t + T))

        # "Forecasts" for exogenous data (here, perfect forecasting)
        temp_oa = batch.temp_oa[:, wrapped_index].unsqueeze(axis=-1).to(device)
        q_solar = batch.q_solar[:, wrapped_index, :].to(device)

        comfort_min = batch.comfort_min[:,
                                        wrapped_index].unsqueeze(-1).to(device)
        comfort_max = batch.comfort_max[:,
                                        wrapped_index].unsqueeze(-1).to(device)

        # I keep the term "tou", but it means electricity price in general.
        batch_predicted_price = batch.predicted_energy_price[:, wrapped_index]
        tou = to_torch(batch_predicted_price).unsqueeze(-1).to(device)

        if self.dr_program.program_type == 'PC':
            power_limit = torch.tensor(
                self.dr_program.power_limit.values[wrapped_index],
                dtype=torch.float32).to(device)
            power_limit = power_limit * torch.ones_like(temp_oa).to(device)
        else:
            power_limit = []

        # Initial conditions
        zone_temp = zone_temp.unsqueeze(axis=1).to(device)
        action_init = action_init.unsqueeze(axis=1).to(device)
        x = x.unsqueeze(axis=1).to(device)

        # In order to get a DPP problem, we need to create auxiliary quantites
        # that pre-multiply parameters.  Use slicing to preserve dimension.
        temp_oa_tou = temp_oa * tou
        action_init_tou = action_init * tou[:, 0:1, 0:1]

        args = [temp_oa, q_solar, zone_temp, action_init, x, tou,
                temp_oa_tou, action_init_tou, comfort_min, comfort_max]
        if self.dr_program.program_type == 'PC':
            args.append(power_limit)

        # TODO: Get the math right here
        if self.use_value_function:
            q_idx = t // self.num_value_interval_steps
            args += [q[:, q_idx].to(device), Q_sqrt[:, :, q_idx].to(device)]

        action = self.cpl.problem(
            *args, solver_args={"solve_method": "ECOS"})[0]
        action = action[:, 0, :]

        return action, {}
