import logging

from lbc.experiments.runner import PolicyRunner, save_runner
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class RLCRunner(PolicyRunner):

    def __init__(self, policy_type: str = None, dr_program: str = None,
                 batch_size: int = None, scenario_config: dict = None,
                 policy_config: dict = None, results_dir: str = None,
                 name_ext: str = None, **kwargs):

        super().__init__(policy_type, dr_program, batch_size,
                         scenario_config, policy_config, results_dir,
                         name_ext, **kwargs)

        self.num_lookahead_steps = self.policy.num_lookahead_steps

    @property
    def name(self):
        la = self.num_lookahead_steps
        return f"RLC-{self.dr_program}-{la}" + self.name_ext

    def run_policy(self, batch_size=None, training=False):

        batch_size = batch_size if batch_size is not None else self.batch_size

        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=batch_size,
            training=training)

        return loss, rollout, meta


def main(**config):
    runner = RLCRunner(**config)
    test_data = runner.run()
    # We can't pickle the rllib trainer, so we delete the policy before
    # saving -- we can always reload it using the rllib interface.
    runner.policy = None
    return save_runner(runner=runner, config=config, test_data=test_data)


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
    parser.add_argument(
        "--lookahead",
        type=int,
        default=2,
        help="number of lookahead steps"
    )
    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = get_config("RLC", **vars(a))
    config["policy_config"]["node_ip_address"] = a.node_ip_address
    config["policy_config"]["num_lookahead_steps"] = a.lookahead
    logger.info(f"CONFIG: {config}")

    _ = main(**config)
