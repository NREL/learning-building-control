import logging

from lbc.experiments.runner import PolicyRunner
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class MPCRunner(PolicyRunner):


    @property
    def name(self):
        la = self.policy_config["num_lookahead_steps"]
        return f"MPC-{self.dr_program}-{la}"

    def run_policy(self, batch_size=None, training=False):
        batch_size = batch_size if batch_size is not None else self.batch_size
        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=batch_size,
            training=training, use_tqdm=True)
        return loss, rollout, meta


def main(**kwargs):
    runner = MPCRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import get_parser
    from lbc.experiments.config import get_config

    parser = get_parser()
    parser.add_argument(
        "--lookahead",
        type=int,
        default=2,
        help="number of lookahead steps"
    )
    parser.add_arugment(
        "--tee",
        action="store_true",
        help="turn on solver logging"
    )
    a = parser.parse_args()

    config = get_config("RLC", **vars(a))
    config["policy_config"]["tee"] = a.tee
    config["policy_config"]["num_lookahead_steps"] = a.lookahead
    print("CONFIG:", config)

    _ = main(**config)
