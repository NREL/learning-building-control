import logging

from lbc.experiments.runner import PolicyRunner
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class RLCRunner(PolicyRunner):

    @property
    def name(self):
        return f"RLC-{self.dr_program}" + self.name_ext

    def run_policy(self, batch_size=None, training=False):

        batch_size = batch_size if batch_size is not None else self.batch_size

        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=batch_size,
            training=training)

        return loss, rollout, meta


def main(**kwargs):
    runner = RLCRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import get_parser
    from lbc.experiments.config import get_config

    parser = get_parser()
    parser.add_argument(
        "--node-ip-address",
        type=str,
        default=None,
        help="node IP address for ray (needed if running with VPN)"
    )
    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = get_config("RLC", **vars(a))
    config["policy_config"]["node_ip_address"] = a.node_ip_address
    print("CONFIG:", config)

    _ = main(**config)
