import logging

from lbc.experiments.runner import PolicyRunner, SCENARIO_DEFAULT
from lbc.experiments.runner import SCENARIO_TEST
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class MPCOneShotRunner(PolicyRunner):
    def run_policy(self):
        # Run the policy on the evaluation set
        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, 
            batch_size=self.batch_size, training=False)
        return loss, rollout, meta


def main(**kwargs):
    runner = MPCOneShotRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import parser

    parser.add_arugment(
        "--tee",
        action="store_true",
        help="turn on solver logging"
    )
    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = {
        "name": f"MPCOneShot-{a.dr_program}",
        "policy_type": "MPCOneShot",
        "dr_program": a.dr_program,
        "batch_size": a.batch_size,
        "scenario_config": SCENARIO_TEST if a.dry_run else SCENARIO_DEFAULT,
        "policy_config": {"tee": a.tee},
        "dry_run": a.dry_run,
        "results_dir": a.results_dir
    }
    print("ARGS:", config)

    _ = main(**config)
