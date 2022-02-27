import logging

from lbc.experiments.runner import PolicyRunner, save_runner
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class MPCOneShotRunner(PolicyRunner):


    @property
    def name(self):
        return f"MPCOneShot-{self.dr_program}" + self.name_ext
        

    def run_policy(self, batch_size=None, training=False):

        batch_size = batch_size if batch_size is not None else self.batch_size

        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=batch_size,
            training=training)

        return loss, rollout, meta


def main(**config):
    
    runner = MPCOneShotRunner(**config)
    test_data = runner.run(training=config.get("training"))
    
    return save_runner(runner=runner, config=config, test_data=test_data)


if __name__ == "__main__":

    from lbc.experiments.runner import get_parser
    from lbc.experiments.config import get_config

    parser = get_parser()
    parser.add_argument(
        "--tee",
        action="store_true",
        help="turn on solver logging"
    )
    parser.add_argument(
        "--control-variance-penalty", 
        type=float,
        default=0.,
        help="penalty weight for action variance, higher -> more smooth"
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="run MPC against the training scenarios (rather than the test set)"
    )

    # Use the args to construct a full configuration for the experiment.
    config = get_config("MPCOneShot", **vars(a))
    config["policy_config"]["tee"] = a.tee
    config["scenario_config"]["control_variance_penalty"] = a.control_variance_penalty
    logger.info(f"CONFIG: {config}")

    _ = main(**config)
