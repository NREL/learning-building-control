import argparse
import logging
import os
import pickle
import time
from typing import Tuple

from lbc.demand_response import DemandResponseProgram
from lbc.policies import (
    Policy, RBCPolicy, MPCOneShotPolicy, MPCPolicy,
    DPCPolicy, CPLPolicy, RLCPolicy)
from lbc.scenario import Scenario


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")

# Configuration dicts for scenarios.  Use _TEST for short runs.
SCENARIO_DEFAULT = {}
SCENARIO_TEST = {
    "start_time": "11:00:00",
    "end_time": "13:00:00",
    "zone_temp_init_mean": 23.0
}

# Policy mapping
POLICY_MAP = {
    "RBC": RBCPolicy,
    "MPCOneShot": MPCOneShotPolicy,
    "MPC": MPCPolicy,
    "DPC": DPCPolicy,
    "CPL": CPLPolicy,
    "RLC": RLCPolicy
}


class PolicyRunner:

    def __init__(
        self,
        name: str = None,
        policy_type: str = None,
        dr_program: str = None,
        batch_size: int = None,
        scenario_config: dict = None,
        policy_config: dict = None,
        training: bool = None,
        dry_run: bool = None
    ):

        assert policy_type in POLICY_MAP, f"invalid policy {policy_type}"

        self.name = name
        self.policy_type = policy_type
        self.dr_program = dr_program
        self.training = training
        self.dry_run = dry_run

        scenario_config["dr_program"] = DemandResponseProgram(dr_program)
        self.scenario_config = scenario_config
        self.scenario = Scenario(**scenario_config)

        # Get the actual batch size
        _batch = self.scenario.make_batch(batch_size, training=training)
        self.batch_size = _batch.q_solar.shape[0]

        # Create a one-off policy config that can be used in the constructor.
        self.policy_config = policy_config

    @classmethod
    def run_policy(self, policy: Policy) -> Tuple[any, any, any]:
        """Do something here and return these values to save."""
        # return loss, rollout, meta
        pass

    def run(self):
        # Instantiate the policy
        policy_cls = POLICY_MAP[self.policy_type]
        policy = policy_cls(scenario=self.scenario, **self.policy_config)

        # Run the policy
        tic = time.time()
        loss, rollout, meta = self.run_policy(policy)
        cpu_time = time.time() - tic

        # Save the results
        self.save(rollout, meta, loss, cpu_time)

    def save(
        self,
        rollout,
        meta,
        batched_loss,
        cpu_time,
    ):

        loss = batched_loss.mean().item()

        logger.info(f"[{self.name}] bsz={self.batch_size},"
                    + f" loss={loss:1.3f}, time={cpu_time:1.1f}")

        if self.dry_run is not True:
            filename = os.path.join(RESULTS_DIR, self.name + ".p")
            with open(filename, "wb") as f:
                pickle.dump(
                    {
                        "name": self.name,
                        "scenario_config": self.scenario_config,
                        "batch_size": self.batch_size,
                        "meta": meta,
                        "rollout": rollout,
                        "cpu_time": cpu_time,
                        "batched_loss": batched_loss,
                        "mean_loss": loss,
                        "policy_config": self.policy_config
                    }, f
                )
            logger.info(f"saved to {filename}")
        else:
            logger.info("dry_run, skipped saving output")


# Common arg parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dr",
    type=str,
    default="TOU",
    dest="dr_program",
    choices=["TOU", "PC", "RTP"]
)
parser.add_argument(
    "--bsz",
    default=1,
    dest="batch_size",
    type=int,
)
parser.add_argument(
    "--dry-run",
    type=int,
    default=0,
    help="0=save, 1=no save"
)
