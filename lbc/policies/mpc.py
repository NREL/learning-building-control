import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pyomo.environ as pyo
from pyomo.environ import value

from pyutilib import subprocess

from lbc.scenario import Scenario, Batch
from lbc.policies import Policy
from lbc.utils import to_torch


subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

logger = logging.getLogger(__file__)


class MPC:
    """MPC controller class that solves a problem instance."""

    def __init__(
        self,
        scenario: Scenario = None,
        step_idx: int = None,
        num_lookahead_steps: int = None,
        temp_oa: np.ndarray = None,
        q_solar: np.ndarray = None,
        zone_temp_init: np.ndarray = None,
        x_init: np.ndarray = None,
        action_init: np.ndarray = None,
        linearize: bool = False,
        wrap_horizon: bool = False,
        solver: str = "ipopt",
        bilinear_eps: float = 1e-2,
        energy_price: np.ndarray = None,
        **kwargs
    ):

        self.scenario = scenario
        self.zone_model = scenario.zone_model
        self.dr_program = scenario.dr_program
        self.step_idx = step_idx
        self.num_lookahead_steps = num_lookahead_steps
        self.temp_oa = np.array(temp_oa).squeeze()
        self.q_solar = np.array(q_solar).squeeze()
        self.zone_temp_init = np.array(zone_temp_init).squeeze()
        self.x_init = np.array(x_init).squeeze()
        self.action_init = np.array(action_init).squeeze()
        self.linearize = linearize
        self.solver = solver
        self.wrap_horizon = wrap_horizon
        self.bilinear_eps = bilinear_eps
        self.energy_price = energy_price

        if linearize is True and action_init is None:
            raise ValueError("Using linearized model requires initial"
                             " actions current operating point)")

        self.solution = None       # Will put the output from solve here

        # Get the action space bounds for use in action constraints
        self.action_min = scenario.action_min
        self.action_max = scenario.action_max

        # Comfort constraint limits
        self.comfort_min = scenario.comfort_min
        self.comfort_max = scenario.comfort_max

        # Private convenience variables
        self.num_zones = len(self.zone_model["A"])
        self.zone_neighbors = self.zone_model["u_neighbor"]

        self.mpc = self.create_model()

    def create_model(self) -> pyo.ConcreteModel:

        # Convenience variables
        T = self.num_lookahead_steps
        N = len(self.zone_model["A"])
        zm = self.zone_model
        zone_temp_init = self.zone_temp_init
        x_init = self.x_init
        action_init = self.action_init
        step_idx = self.step_idx
        total_rows = self.scenario.num_episode_steps
        nbrs = self.zone_model["u_neighbor"]

        # Handle requested wrapping strategy
        if self.wrap_horizon:
            # If wrapping, use the full lookahead and wrap indices into
            # exogenous data
            wrapped_index = [x % total_rows for x in range(step_idx,
                                                           step_idx + T)]
        else:
            # Otherwise, shorten the lookahead to end with the data
            T = min(T, total_rows - step_idx)
            wrapped_index = list(range(step_idx, step_idx + T))

        m = pyo.ConcreteModel()

        # INDEX SETS
        m.time = pyo.RangeSet(0, T-1)
        m.time_not_init = pyo.RangeSet(1, T-1)
        m.time_control = pyo.RangeSet(0, T-1)
        m.time_terminal = T
        m.zone = pyo.RangeSet(0, N-1)
        m.control = pyo.RangeSet(0, N)
        m.feature = pyo.RangeSet(0, 3)

        # VARIABLES
        m.zone_temp = pyo.Var(m.time, m.zone, domain=pyo.Reals)
        m.x = pyo.Var(m.time, m.zone, domain=pyo.Reals)
        m.u = pyo.Var(m.time_control, m.zone, m.feature, domain=pyo.Reals)
        m.action = pyo.Var(m.time_control, m.control, domain=pyo.Reals)
        m.fan_power = pyo.Var(m.time, domain=pyo.Reals)
        m.chiller_power = pyo.Var(m.time, domain=pyo.Reals)
        m.p_consumed = pyo.Var(m.time, domain=pyo.NonNegativeReals)
        m.comfort_viol_lower = pyo.Var(m.time, m.zone,
                                       domain=pyo.NonNegativeReals)
        m.comfort_viol_upper = pyo.Var(m.time, m.zone,
                                       domain=pyo.NonNegativeReals)

        # PARAMETERS
        # Exogenous data.
        temp_oa = self.temp_oa[wrapped_index]
        m.temp_oa = pyo.Param(m.time,
                              initialize={t: temp_oa[t] for t in m.time})

        q_solar = self.q_solar[:, wrapped_index]
        m.q_solar = pyo.Param(
            m.time, m.zone,
            initialize={(t, z): q_solar[z, t]
                        for z in m.zone for t in m.time})

        comfort_min = self.scenario.comfort_min[wrapped_index]
        m.comfort_min = pyo.Param(
            m.time,
            initialize={t: comfort_min[t] for t in m.time}
        )

        comfort_max = self.scenario.comfort_max[wrapped_index]
        m.comfort_max = pyo.Param(
            m.time,
            initialize={t: comfort_max[t] for t in m.time}
        )

        # CONSTRAINTS
        # Bound constraints
        m.action_lb_cons = pyo.Constraint(
            m.time_control, m.control,
            rule=lambda m, t, c: m.action[t, c] >= self.action_min[c])

        m.action_ub_cons = pyo.Constraint(
            m.time_control, m.control,
            rule=lambda m, t, c: m.action[t, c] <= self.action_max[c])

        # Definitions.
        m.t_oa_cons = pyo.Constraint(
            m.time_control, m.zone,
            rule=lambda m, t, z:
                m.u[t, z, 0] == m.temp_oa[t] - m.zone_temp[t, z])

        # Q cooling constraints
        if self.linearize:
            # Otherwise, we expand around the current operating point
            m.q_cooling_cons = pyo.Constraint(
                m.time_control, m.zone,
                rule=lambda m, t, z:
                    m.u[t, z, 1] == (m.action[t, z] - action_init[z])
                    * (action_init[N] - zone_temp_init[z])
                    + action_init[z] * (m.action[t, N] - m.zone_temp[t, z]))
        else:
            # # In full nonlinear model, this constraint always holds
            m.q_cooling_cons = pyo.Constraint(
                m.time_control, m.zone,
                rule=lambda m, t, z:
                    m.u[t, z, 1] == m.action[t, z] * (m.action[t, N]
                                                      - m.zone_temp[t, z]))

        # This feature is the difference between current zone and the selected
        # neighbor zone.
        m.neighbor_cons = pyo.Constraint(
            m.time_control, m.zone,
            rule=lambda m, t, z:
                m.u[t, z, 2] == m.zone_temp[t, nbrs[z]] - m.zone_temp[t, z])

        # Q solar definition.
        m.q_solar_cons = pyo.Constraint(
            m.time_control, m.zone,
            rule=lambda m, t, z: m.u[t, z, 3] == m.q_solar[t, z])

        # Total fan power per time step.
        hvac_cop, fan_coeff_1, fan_coeff_2 = self.zone_model['hvac_parameters']
        m.fan_power_cons = pyo.Constraint(
            m.time_control,
            rule=lambda m, t:
                m.fan_power[t] == (
                    fan_coeff_1 * sum(m.action[t, z] for z in m.zone) ** 3
                    + fan_coeff_2))

        # P_consumed is the fan power plus q_hvac
        if self.linearize:
            m.p_consumed_cons = pyo.Constraint(
                m.time_control, rule=lambda m, t:
                    m.p_consumed[t] == m.fan_power[t]
                    + sum(m.action[t, z] for z in m.zone) * m.temp_oa[t]
                    - sum((m.action[t, z] - action_init[z]) * action_init[N]
                          + action_init[z] * m.action[t, N] for z in m.zone))
        else:
            m.chiller_power_cons = pyo.Constraint(
                m.time_control, rule=lambda m, t:
                    m.chiller_power[t] == hvac_cop
                    * sum(m.action[t, z] for z in m.zone)
                    * (m.temp_oa[t] - m.action[t, N])
            )
            m.p_consumed_cons = pyo.Constraint(
                m.time_control, rule=lambda m, t:
                    m.p_consumed[t] == m.fan_power[t] + m.chiller_power[t])

        # Zone temp comfort violations
        m.comfort_viol_lower_cons = pyo.Constraint(
            m.time, m.zone,
            rule=lambda m, t, z:
                m.zone_temp[t, z]
                + m.comfort_viol_lower[t, z] >= m.comfort_min[t])
        m.comfort_viol_upper_cons = pyo.Constraint(
            m.time, m.zone,
            rule=lambda m, t, z:
                m.zone_temp[t, z]
                - m.comfort_viol_upper[t, z] <= m.comfort_max[t])

        # Dynamics.  Need rule functions here to handle initial conditions.

        # x: state variable which (as far as I can tell) is a rescaling of the
        # zone temperatures (?)
        m.x_init_cons = pyo.Constraint(m.zone,
                                       rule=lambda m, z:
                                       m.x[0, z] == x_init[z])

        def x_rule(m, t, z):
            return m.x[t, z] == zm["A"][z] * m.x[t-1, z] \
                + sum([zm["B"][i, z] * m.u[t-1, z, i] for i in m.feature])
        m.x_cons = pyo.Constraint(m.time_not_init, m.zone, rule=x_rule)

        # Zone temperature dynamics.
        def zone_temp_rule(m, t, z):
            if t == 0:
                return m.zone_temp[0, z] == zone_temp_init[z]
            else:
                return m.zone_temp[t, z] ==\
                    zm["C"][z] * m.x[t, z] +\
                    zm["mean_output"][z]
        m.zone_temp_cons = pyo.Constraint(m.time, m.zone, rule=zone_temp_rule)

        if self.dr_program.program_type == 'PC':
            m.power_viol = pyo.Var(m.time, domain=pyo.NonNegativeReals)
            m.power_viol_cons = pyo.Constraint(
                m.time,
                rule=lambda m, t: m.p_consumed[t] - m.power_viol[t]
                <= self.dr_program.power_limit.values[wrapped_index[t]][0])

        delta_t = self.zone_model['delta_t']

        def cost_rule(m):
            _idx = wrapped_index
            power_cost = sum(self.energy_price[_idx[t]]
                             * m.p_consumed[t] for t in m.time) * delta_t
            discomfort_cost = self.scenario.comfort_penalty * sum(
                m.comfort_viol_lower[t, z]**2 + m.comfort_viol_upper[t, z]**2
                for t in m.time for z in m.zone)
            cost = power_cost + discomfort_cost

            if self.dr_program.program_type == 'PC':
                power_violation_cost = self.dr_program.pc_penalty * sum(
                    m.power_viol[t]**2 for t in m.time)
                cost += power_violation_cost
            return cost

        # Set the cost, using the supplied objective if available
        m.cost = pyo.Objective(rule=cost_rule)

        return m

    def solve(self, tee=False) -> pd.DataFrame:
        """Returns a dataframe with the solution params and vars."""
        opt = SolverFactory(self.solver)
        try:
            self.solution = opt.solve(self.mpc, tee=tee)
            return self.parse()
        except Exception as e:
            logger.error("solve failed: {}".format(e))

    def parse(self) -> pd.DataFrame:
        """Returns a pandas dataframe containing all model params and vars.
        Assumes that the solve method has already been called.
        """

        if self.solution is None:
            logger.error("No solution, did you call solve first?")
            return None

        if self.solution.solver.status != SolverStatus.ok:
            logger.error("solver status: ", self.solution.solver.status)
        if self.solution.solver.termination_condition \
                != TerminationCondition.optimal:
            logger.warning("termination condition: ",
                           self.solution.solver.termination_condition)

        data = {}

        for zone in range(self.num_zones):

            key = "flow_{}".format(zone)
            data[key] = [value(self.mpc.action[t, zone])
                         for t in self.mpc.time]

            key = "zone_temp_{}".format(zone)
            data[key] = [value(self.mpc.zone_temp[t, zone])
                         for t in self.mpc.time]

        key = "discharge_temp"
        data[key] = [value(self.mpc.action[t, self.num_zones])
                     for t in self.mpc.time]

        # Sort the keys so the dataframe looks nice
        data = {k: data[k] for k in sorted(data.keys())}
        df = pd.DataFrame(data)

        return df


class MPCPolicy(Policy):

    def __init__(
        self,
        num_lookahead_steps: int = 4,
        linearize: bool = False,
        one_shot: bool = False,
        solver: str = "ipopt",
        **kwargs,
    ):
        super().__init__()

        self.num_lookahead_steps = num_lookahead_steps
        self.linearize = linearize
        self.one_shot = one_shot
        self.solver = solver

        self.cached_actions = None
        self.cached_dfs = []

    def __call__(
        self,
        scenario: Scenario,
        batch: Batch,
        t: int,
        x: torch.tensor,
        zone_temp: torch.tensor,
        action_init: torch.tensor = None,
        training: bool = True,
        **kwargs
    ) -> Tuple[torch.tensor, dict]:

        bsz, _, _ = batch.q_solar.shape

        # Lookahead over whole horizon if one-shotting.
        la_steps = self.num_lookahead_steps \
            if not self.one_shot else scenario.num_episode_steps
        use_cached_actions = False
        if self.one_shot and self.cached_actions is not None:
            use_cached_actions = True

        if self.linearize:
            if t == 0:
                mdot = scenario.action_min[:-1].tolist()
                dtemp = [scenario.action_max[-1]]
                action_init = np.array(mdot + dtemp)
                action_init = torch.from_numpy(
                    action_init) * torch.ones(bsz, *action_init.shape)
            else:
                assert action_init is not None,\
                    "action_init must be provided for linearization"
                action_init = action_init.detach().numpy()

        # Convert batch tensors to numpy arrays
        temp_oa = batch.temp_oa.detach().numpy()
        q_solar = batch.q_solar.detach().numpy()
        if self.one_shot:
            batch_energy_price = batch.energy_price.detach().numpy()
        else:
            batch_energy_price = batch.predicted_energy_price.detach().numpy()
        x = x.detach().numpy()
        zone_temp = zone_temp.detach().numpy()

        torch_actions = []
        for b in range(bsz):

            if self.one_shot and self.cached_actions is None:
                logger.info(f"one-shot: batch sample ({b+1}/{bsz})")

            if use_cached_actions:

                return self.cached_actions[:, t, :], {"df": self.cached_dfs}

            else:

                # Get the data for each sample.
                _temp_oa = temp_oa[b, :].squeeze()
                _q_solar = q_solar[b, :, :].squeeze().T
                _zone_temp = zone_temp[b, :].squeeze()
                _x = x[b, :].squeeze()
                _action_init = action_init[b, :].squeeze() \
                    if action_init is not None else None

                energy_price = batch_energy_price[b, :].squeeze()

                # Create the MPC model instance.
                mpc = MPC(
                    zone_model=scenario.zone_model,
                    scenario=scenario, step_idx=t,
                    num_lookahead_steps=la_steps,
                    temp_oa=_temp_oa, q_solar=_q_solar,
                    zone_temp_init=_zone_temp, x_init=_x,
                    action_init=_action_init, linearize=self.linearize,
                    solver=self.solver, bilinear_eps=1,
                    energy_price=energy_price)

                # Solve the MPC problem and extract the first action.
                df = mpc.solve()
                action = df[["flow_0", "flow_1", "flow_2", "flow_3", "flow_4",
                             "discharge_temp"]]

                if self.one_shot:
                    action = action.values
                    self.cached_dfs.append(df.copy())
                else:
                    action = action.values[0, :].squeeze()

            torch_actions.append(to_torch(action))

        # Create torch output of shape [batch_size, *action.shpae]
        torch_actions = torch.stack(torch_actions)
        torch_actions = torch_actions.reshape(bsz, *action.shape)

        # If you are here in the one_shot workflow, you haven't yet cached
        # actions.
        return_actions = torch_actions
        if self.one_shot:
            self.cached_actions = torch_actions.detach().clone()
            return_actions = self.cached_actions[:, t, :].squeeze()

        return return_actions, {"df": self.cached_dfs}


class MPCOneShotPolicy(Policy):

    def __init__(
        self,
        linearize: bool = False,
        solver: str = "ipopt",
        **kwargs
    ):
        super().__init__()

        self.linearize = linearize
        self.mpc = MPCPolicy(linearize=linearize, one_shot=True, solver=solver)

    def __call__(self, *args, **kwargs) -> Tuple[torch.tensor, dict]:

        return self.mpc(*args, **kwargs)
