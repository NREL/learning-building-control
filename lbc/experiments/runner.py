import argparse
import logging
import os
import pickle
import time
from typing import Tuple

from lbc.demand_response import DemandResponseProgram
from lbc.policies import (
    RBCPolicy, MPCOneShotPolicy, MPCPolicy, DPCPolicy, CPLPolicy, RLCPolicy)
from lbc.scenario import Scenario
from lbc.experiments.config import DEFAULT_RESULTS_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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
        policy_type: str = None,
        dr_program: str = None,
        batch_size: int = None,
        scenario_config: dict = None,
        policy_config: dict = None,
        results_dir: str = None,
        name_ext: str = None,
        **kwargs
    ):

        assert policy_type in POLICY_MAP, f"invalid policy {policy_type}"

        self.policy_type = policy_type
        self.dr_program = dr_program
        self.results_dir = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
        self.name_ext = f"-{name_ext}" if name_ext is not None else ""

        scenario_config["dr_program"] = DemandResponseProgram(dr_program)
        self.scenario_config = scenario_config
        self.scenario = Scenario(**scenario_config)

        # Get the actual batch size
        _batch = self.scenario.make_batch(batch_size)
        self.batch_size = _batch.q_solar.shape[0]

        # Create a one-off policy config that can be used in the constructor.
        self.policy_config = policy_config

        # Instantiate the policy
        policy_cls = POLICY_MAP[self.policy_type]
        self.policy = policy_cls(scenario=self.scenario, **self.policy_config)


    @classmethod
    def run_policy(
        self,
        batch_size: int = None,
        training: bool = False,
        save: bool = True
    ) -> Tuple[any, any, any]:
        """Do something here and return these values to save."""
        # return loss, rollout, meta
        pass


    @classmethod
    def train_policy(self):
        raise NotImplementedError


    @property
    def name(self):
        raise NotImplementedError


    def run(
        self,
        batch_size: int = None,
        training: bool = False,
        **kwargs
    ) -> Tuple[any, any, any]:

        # Run the policy
        tic = time.time()
        loss, rollout, meta = self.run_policy(
            batch_size=batch_size, training=training)
        meta["run_time"] = time.time() - tic

        # Save the results
        #self.save(rollout, meta, loss, cpu_time)

        return loss, rollout, meta


    # def save(
    #     self,
    #     rollout,
    #     meta,
    #     batched_loss,
    #     cpu_time,
    #     name_suffix = None
    # ):

    #     loss = batched_loss.mean().item()

    #     logger.info(f"[{self.name}] bsz={self.batch_size},"
    #                 + f" loss={loss:1.3f}, time={cpu_time:1.1f}")

    #     # Make the output directory if it doesn't exist.
    #     pathlib.Path(self.results_dir).mkdir(parents=True, exist_ok=True) 

    #     # Save the output
    #     name = self.name if name_suffix is None else self.name + "-" + name_suffix
    #     filename = os.path.join(self.results_dir, name + ".p")
    #     with open(filename, "wb") as f:
    #         pickle.dump(
    #             {
    #                 "name": self.name,
    #                 "scenario_config": self.scenario_config,
    #                 "batch_size": self.batch_size,
    #                 "meta": meta,
    #                 "rollout": rollout,
    #                 "cpu_time": cpu_time,
    #                 "batched_loss": batched_loss,
    #                 "loss": loss,
    #                 "policy_config": self.policy_config
    #             }, f
    #         )
    #     logger.info(f"saved to {filename}")


def save_runner(runner, config, test_data, train_data=None, **kwargs):

    payload = {
        "runner": runner,
        "config": config,
        "test_data": test_data,
        "train_data": train_data,
        **kwargs
    }
    filename = os.path.join(runner.results_dir, runner.name + ".p")

    with open(filename, "wb") as f:
        pickle.dump(payload, f)    
    logger.info(f"saved to {filename}")

    return payload


def get_parser():
    # Common arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="directory to save output in"
    )
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
        help="0=full scenario, 1=short scenairo"
    )
    parser.add_argument(
        "--name-ext",
        type=str,
        default=None,
        help="extension to add to output filename"
    )
    return parser
