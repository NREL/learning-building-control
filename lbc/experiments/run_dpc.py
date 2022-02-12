from copy import deepcopy
import logging

import numpy as np
from tqdm import tqdm

import torch

from lbc.experiments.runner import PolicyRunner, SCENARIO_DEFAULT
from lbc.experiments.runner import SCENARIO_TEST
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class DPCRunner(PolicyRunner):

    def train_policy(self):

        policy = self.policy

        opt = torch.optim.Adam(
            policy.model.parameters(), lr=self.policy_config["lr"])

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt, factor=0.5, patience=10, cooldown=10)
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=np.sqrt(.1))

        losses = []
        test_losses = []
        pbar = tqdm(range(self.policy_config["num_epochs"]))
        for _ in pbar:
            try:
                # Simulate on the training set and perform gradient update
                loss, rollout, meta = simulate(
                    policy=policy, scenario=self.scenario, batch_size=self.batch_size,
                    training=True)

                loss = loss.mean()

                # Normalize the loss function by number steps so we can
                # stably handle different episode lengths with the same lr.
                opt_loss = loss / self.scenario.num_episode_steps

                # Gradient update.
                opt.zero_grad()
                opt_loss.backward()
                opt.step()

                # # Evaluate on the test set for monitoring
                test_loss, _, _ = simulate(
                    policy=policy, scenario=self.scenario, batch_size=self.batch_size,
                    training=False)
                test_loss = test_loss.mean()

                # Track the training and test losses.
                losses.append(loss.detach().numpy())
                test_losses.append(test_loss.detach().numpy())

                pbar.set_description(
                    f"{losses[-1]:1.3f}, {test_losses[-1]:1.3f},")
                    
            except KeyboardInterrupt:
                logger.info("stopped")
                break

        meta.update({
            "model": deepcopy(policy.model),
            "losses": losses,
            "test_losses": test_losses
        })

        return loss, rollout, meta


    def run_policy(self):
        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=self.batch_size,
            training=False)
        return loss, rollout, meta


def main(**kwargs):
    runner = DPCRunner(**kwargs)
    runner.run()


if __name__ == "__main__":

    from lbc.experiments.runner import parser

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
    parser.add_argument(
        "--num-intervals",
        type=int,
        default=48,
        help="number of time embeddings to use in policy model"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="dimension of hidden layers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="torch device"
    )
    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = {
        "name": f"DPC-{a.dr_program}-{a.hidden_dim}-{a.device}",
        "policy_type": "DPC",
        "batch_size": a.batch_size,
        "dr_program": a.dr_program,
        "scenario_config": SCENARIO_TEST if a.dry_run else SCENARIO_DEFAULT,
        "policy_config": {
            "model_config": {
                "hidden_dim": a.hidden_dim,
                "num_time_windows": a.num_time_windows
            },
            "lr": a.lr,
            "num_epochs": a.num_epochs,
        },
        "dry_run": a.dry_run,
        "results_dir": a.results_dir
    }
    print("ARGS:", config)

    _ = main(**config)

