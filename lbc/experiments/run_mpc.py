import logging

from lbc.experiments.runner import PolicyRunner, SCENARIO_DEFAULT
from lbc.experiments.runner import SCENARIO_TEST
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class MPCRunner(PolicyRunner):
    def run_policy(self, policy):
        loss, rollout, meta = simulate(
            policy=policy, scenario=self.scenario, batch_size=self.batch_size,
            use_tqdm=True)
        return loss, rollout, meta


def main(**kwargs):
    runner = MPCRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import parser

    parser.add_argument(
        "--lookahead",
        type=int,
        default=2,
        help="number of lookahead steps"
    )
    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = {
        "name": f"MPC-{a.dr_program}-{a.lookahead}",
        "policy_type": "MPC",
        "batch_size": a.batch_size,
        "dr_program": a.dr_program,
        "scenario_config": SCENARIO_TEST if a.dry_run else SCENARIO_DEFAULT,
        "policy_config": {"num_lookahead_steps": a.lookahead},
        "training": False,
        "dry_run": a.dry_run,
        "results_dir": a.results_dir
    }
    print("ARGS:", config)

    _ = main(**config)
