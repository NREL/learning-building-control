import logging

from lbc.experiments.runner import PolicyRunner, SCENARIO_DEFAULT
from lbc.experiments.runner import SCENARIO_TEST
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


# Default setpoints for each policy
SETPOINTS = {
    "TOU": [(0, 27), (83, 24), (131, 21), (143, 24), (215, 24), (215, 27)],
    "PC": [(0, 27), (83, 24), (143, 21), (155, 24), (203, 24), (215, 27)],
    "RTP": [(0, 27), (83, 24), (131, 21), (143, 24), (215, 27)]
}


class RBCRunner(PolicyRunner):
    def run_policy(self):
        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=self.batch_size,
            training=False)
        return loss, rollout, meta


def main(**kwargs):
    runner = RBCRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import parser

    parser.add_argument(
        "--p-flow",
        type=float,
        default=1.0,
        help="ratio of total zone flow applied by controller"
    )
    parser.add_argument(
        "--p-temp",
        type=float,
        default=0.8,
        help="ratio of total discharge temp applied by controller"
    )
    parser.add_argument(
        "--band-width",
        type=float,
        default=1.,
        help="width of comfort band"
    )

    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = {
        "name": f"RBC-{a.dr_program}-{a.p_flow:1.2f}-{a.p_temp:1.2f}",
        "policy_type": "RBC",
        "batch_size": a.batch_size,
        "dr_program": a.dr_program,
        "scenario_config": SCENARIO_TEST if a.dry_run else SCENARIO_DEFAULT,
        "policy_config": {
            "setpoints": SETPOINTS[a.dr_program],
            "p_flow": a.p_flow,
            "p_temp": a.p_temp
        },
        "training": False,
        "dry_run": a.dry_run,
        "results_dir": a.results_dir
    }
    print("ARGS:", config)

    _ = main(**config)
