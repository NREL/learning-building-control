import logging

from lbc.experiments.runner import PolicyRunner, SCENARIO_DEFAULT
from lbc.experiments.runner import SCENARIO_TEST
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class RLCRunner(PolicyRunner):

    def run_policy(self, policy):
        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=self.batch_size,
            trainig=False)
        return loss, rollout, meta


def main(**kwargs):
    runner = RLCRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import parser

    parser.add_argument(
        "--node-ip-address",
        type=str,
        default=None,
        help="node IP address for ray (needed if running with VPN)"
    )
    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = {
        "name": f"RLC-{a.dr_program}",
        "policy_type": "RLC",
        "dr_program": a.dr_program,
        "batch_size": a.batch_size,
        "scenario_config": SCENARIO_TEST if a.dry_run else SCENARIO_DEFAULT,
        "policy_config": {
            "node_ip_address": a.node_ip_address
        },
        "training": False,
        "dry_run": a.dry_run,
        "results_dir": a.results_dir
    }
    print("ARGS:", config)

    _ = main(**config)
