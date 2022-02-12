import logging

import numpy as np

from tqdm import tqdm

import torch

from lbc.experiments.runner import PolicyRunner, SCENARIO_DEFAULT
from lbc.experiments.runner import SCENARIO_TEST
from lbc.simulate import simulate


logger = logging.getLogger(__name__)


class CPLRunner(PolicyRunner):

    def train_policy(self):

        policy = self.policy

        use_value_function = self.policy_config["use_value_function"]
        num_epochs = self.policy_config["num_epochs"]
        num_episode_steps = self.scenario.num_episode_steps
        batch_size = self.batch_size

        # Initialize the value function tensors
        q = torch.zeros(
            (5, policy.num_time_windows), dtype=torch.float32, requires_grad=True)
        Q_sqrt = torch.zeros(
            (5, 5, policy.num_time_windows), dtype=torch.float32, requires_grad=True)

        opt = torch.optim.Adam([q, Q_sqrt], lr=self.policy_config["lr"])

        # If not learning the value function, we'll just run once against the
        # test set.
        num_epochs = num_epochs if use_value_function else 1

        # Main loop
        losses = []
        test_losses = []
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:

            try:

                # If learning the value function, run simulation and do
                # gradient update
                loss, rollout, meta = simulate(
                    policy=policy, scenario=self.scenario, batch_size=self.batch_size, 
                    training=True, q=q, Q_sqrt=Q_sqrt)

                # Take mean and normalize
                loss = loss.mean()
                opt_loss = loss / num_episode_steps

                # Gradient step
                opt.zero_grad()
                opt_loss.backward()
                opt.step()

                # Evaluate on the test set
                test_loss, _, _ = simulate(
                    policy=policy, scenario=self.scenario, batch_size=batch_size,
                    training=False, q=q, Q_sqrt=Q_sqrt)
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
            "model": [q.clone().detach(), Q_sqrt.clone().detach()],
            "losses": losses,
            "test_losses": test_losses
        })

        return loss, rollout, meta


    def run_policy(self):
        policy = self.policy
        loss, rollout, meta = simulate(
            policy=policy, scenario=self.scenario, batch_size=self.batch_size,
            training=False)
        return loss, rollout, meta


def main(**kwargs):
    runner = CPLRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import parser

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
        default=1,
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

    # Use the args to construct a full configuration for the experiment.
    config = {
        "name": f"CPL-{a.dr_program}-{a.lookahead}-q={a.use_value_function}",
        "policy_type": "CPL",
        "batch_size": a.batch_size,
        "dr_program": a.dr_program,
        "scenario_config": SCENARIO_TEST if a.dry_run else SCENARIO_DEFAULT,
        "policy_config": {
            "lookahead": a.lookahead,
            "lr": a.lr,
            "num_epochs": a.num_epochs,
            "use_value_function": a.use_value_function,
            "num_time_windows": a.num_time_windows,
        },
        "dry_run": a.dry_run,
        "results_dir": a.results_dir
    }
    print("ARGS:", config)

    _ = main(**config)
