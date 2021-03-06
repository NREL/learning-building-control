import logging
import os
import time

from tqdm import tqdm

import torch

from lbc.experiments.runner import PolicyRunner, save_runner
from lbc.simulate import simulate


logger = logging.getLogger(__name__)


class CPLRunner(PolicyRunner):


    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Initialize the value function tensors
        self.q = torch.zeros(
            (5, self.policy.num_time_windows), dtype=torch.float32, requires_grad=True)
        self.Q_sqrt = torch.zeros(
            (5, 5, self.policy.num_time_windows), dtype=torch.float32, requires_grad=True)


    @property
    def name(self):
        la = self.policy_config["lookahead"]
        uvf = self.policy_config["use_value_function"]
        ntw = self.policy_config["num_time_windows"]
        return f"CPL-{self.dr_program}-{la}-{ntw}-{uvf}" + self.name_ext


    def train_policy(self):

        # Convenient references
        policy = self.policy
        use_value_function = self.policy_config["use_value_function"]
        num_epochs = self.policy_config["num_epochs"]
        num_episode_steps = self.scenario.num_episode_steps
        batch_size = self.batch_size

        opt = torch.optim.Adam([self.q, self.Q_sqrt], lr=self.policy_config["lr"])

        # If not learning the value function, we'll just run once against the
        # test set.
        num_epochs = num_epochs if use_value_function else 1

        # Main loop
        losses = []
        test_losses = []
        tic = time.time()
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:

            try:

                # If learning the value function, run simulation and do
                # gradient update
                loss, rollout, meta = simulate(
                    policy=policy, scenario=self.scenario, batch_size=self.batch_size, 
                    training=True, shuffle=True, q=self.q, Q_sqrt=self.Q_sqrt)

                # Take mean and normalize
                loss = loss.mean()
                opt_loss = loss / num_episode_steps

                # Gradient step
                opt.zero_grad()
                opt_loss.backward()
                opt.step()

                # Evaluate on the test set
                test_loss, _, _ = simulate(
                    policy=policy, scenario=self.scenario, batch_size=31,
                    training=False, q=self.q, Q_sqrt=self.Q_sqrt)
                test_loss = test_loss.mean()

                # Update loss traces
                losses.append(loss.detach().numpy())
                test_losses.append(test_loss.detach().numpy())

                pbar.set_description(
                    f"{losses[-1]:1.3f}, {test_losses[-1]:1.3f}")

            except KeyboardInterrupt:
                logger.info("stopped")
                break

        meta.update({
            "model": [self.q.clone().detach(), self.Q_sqrt.clone().detach()],
            "losses": losses,
            "test_losses": test_losses,
            "train_time": time.time() - tic
        })

        # Save results
        #self.save(rollout, meta, loss, cpu_time, name_suffix="train")

        return loss, rollout, meta


    def run_policy(self, batch_size=None, training=False):

        batch_size = batch_size if batch_size is not None else self.batch_size
        
        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=batch_size,
            training=training, q=self.q, Q_sqrt=self.Q_sqrt)
        
        return loss, rollout, meta


def main(**config):
    
    runner = CPLRunner(**config)
    
    train_data = None
    policy_config = config.get("policy_config", {})
    if "use_value_function" in policy_config and policy_config["use_value_function"] == 1:
        logger.info("starting training run")
        train_data = runner.train_policy()

    logger.info("evaluating policy")
    test_data = runner.run(batch_size=31)

    return save_runner(runner=runner, config=config, test_data=test_data, 
        train_data=train_data)



if __name__ == "__main__":

    from lbc.experiments.runner import get_parser
    from lbc.experiments.config import get_config

    parser = get_parser()
    parser.add_argument(
        "--lookahead",
        type=int,
        default=4,
        help="number of lookahead steps"
    )
    parser.add_argument(
        "--use-value-function",
        type=int,
        default=1,
        help="learn a value function (1=yes, 0=no)"
    )
    parser.add_argument(
        "--num-time-windows",
        type=int,
        default=24,
        help="number of time windows to use in modeling value function"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="number of training epochs"
    )
    a = parser.parse_args()

    config = get_config("CPL", **vars(a))
    config["policy_config"] = {
        "lookahead": a.lookahead,
        "lr": a.lr,
        "num_epochs": a.num_epochs,
        "use_value_function": a.use_value_function,
        "num_time_windows": a.num_time_windows,
    }

    logger.info(f"CONFIG: {config}")
    
    _ = main(**config)
