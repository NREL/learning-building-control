import logging

from lbc.experiments.runner import PolicyRunner, save_runner
from lbc.simulate import simulate

logger = logging.getLogger(__file__)



class RBCRunner(PolicyRunner):

    @property
    def name(self):
        pf = self.policy_config["p_flow"]
        pt = self.policy_config["p_temp"]
        return f"RBC-{self.dr_program}-{pf:1.3f}-{pt:1.3f}" + self.name_ext

    def run_policy(self, batch_size=None, training=False):

        batch_size = batch_size if batch_size is not None else self.batch_size
        
        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=batch_size,
            training=training)
        
        return loss, rollout, meta


def main(**config):
    runner = RBCRunner(**config)
    test_data = runner.run()
    return save_runner(runner=runner, config=config, test_data=test_data)
    

if __name__ == "__main__":

    from lbc.experiments.runner import get_parser
    from lbc.experiments.config import get_config

    parser = get_parser()
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
    a = parser.parse_args()

    config = get_config("RBC", **vars(a))
    config["policy_config"]["p_flow"] = a.p_flow
    config["policy_config"]["p_temp"] = a.p_temp
    logger.info(f"CONFIG: {config}") 
      
    _ = main(**config)
