import logging

import numpy as np

from tqdm import tqdm

import torch

from lbc.experiments.runner import PolicyRunner, SCENARIO_DEFAULT
from lbc.experiments.runner import SCENARIO_TEST
from lbc.simulate import simulate


logger = logging.getLogger(__name__)


class CPLRunner(PolicyRunner):

    def run_policy(self, policy):

        use_value_function = self.policy_config["use_value_function"]
        value_interval = self.policy_config["num_value_interval_steps"]
        num_epochs = self.policy_config["num_epochs"]
        batch_size = self.batch_size

        # Initialize the value function tensors
        num_intervals = self.scenario.num_episode_steps // value_interval + 1
        q = torch.zeros(
            (5, num_intervals), dtype=torch.float32, requires_grad=True)
        Q_sqrt = torch.zeros(
            (5, 5, num_intervals), dtype=torch.float32, requires_grad=True)

        opt = torch.optim.Adam([q, Q_sqrt], lr=self.policy_config["lr"])

        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.1)

        # If not learning the value function, we'll just run once against the
        # test set.
        num_epochs = num_epochs if use_value_function else 1

        # Main loop
        best_test_loss = np.inf
        best_model = None
        losses = []
        test_losses = []
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:

            try:

                if use_value_function:
                    # If learning the value function, run simulation and do
                    # gradient update
                    total_loss, _, _ = simulate(
                        policy=policy, scenario=self.scenario,
                        batch_size=self.batch_size, q=q, Q_sqrt=Q_sqrt)

                    loss = total_loss.mean()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    losses.append(loss.detach().numpy())

                else:
                    logger.info(
                        "With use_value_function=False we only run one epoch"
                        + " against the test set. Training loss will be nan"
                        + " here (expected)."
                    )
                    total_loss = None
                    losses.append(np.nan)

                # Evaluate on the test set
                test_total_loss, test_rollout, meta = simulate(
                    policy=policy, scenario=self.scenario,
                    batch_size=min(batch_size, 31),
                    training=False, q=q, Q_sqrt=Q_sqrt)
                test_loss = test_total_loss.mean()
                test_losses.append(test_loss.detach().numpy())

                scheduler.step()

                if test_losses[-1] < best_test_loss:
                    best_test_loss = test_losses[-1]
                    best_model = [q.clone().detach(), Q_sqrt.clone().detach()]

                pbar.set_description(
                    f"{losses[-1]:1.3f}, {test_losses[-1]:1.3f}" \
                    + f" {scheduler._last_lr[0]:1.3e}")

            except KeyboardInterrupt:
                logger.info("stopped")
                break

        meta.update({
            "best_test_loss": best_test_loss,
            "best_model": best_model,
            "losses": losses,
            "test_losses": test_losses
        })

        return best_test_loss, test_rollout, meta


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
        "--num-value-interval-steps",
        type=int,
        default=1,
        help="steps per value function interval"
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
            "num_value_interval_steps": a.num_value_interval_steps,
        },
        "training": bool(a.use_value_function),
        "dry_run": a.dry_run
    }
    print("ARGS:", config)

    _ = main(**config)
