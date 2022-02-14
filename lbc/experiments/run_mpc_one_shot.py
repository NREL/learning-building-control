import logging

from lbc.experiments.runner import PolicyRunner
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class MPCOneShotRunner(PolicyRunner):

    @property
    def name(self):
        return f"MPCOneShot-{self.dr_program}"
        
    def run_policy(self, batch_size=None, training=False):
        batch_size = batch_size if batch_size is not None else self.batch_size
        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=batch_size,
            training=training)
        return loss, rollout, meta


def main(**kwargs):
    runner = MPCOneShotRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import get_parser
    from lbc.experiments.config import get_config

    parser = get_parser()
    parser.add_argument(
        "--tee",
        action="store_true",
        help="turn on solver logging"
    )
    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = get_config("MPCOneShot", **vars(a))
    config["policy_config"]["tee"] = a.tee
    print("CONFIG:", config)

    _ = main(**config)
