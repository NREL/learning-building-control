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

    def run_policy(self, policy):

        opt = torch.optim.Adam(policy.model.parameters(),
                               lr=self.policy_config["lr"])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=10, cooldown=10)

        best_test_loss = np.inf
        best_model = None
        losses = []
        test_losses = []
        pbar = tqdm(range(self.policy_config["num_epochs"]))

        for _ in pbar:
            try:
                # Simulate on the training set and perform gradient update
                total_loss, _, _ = simulate(
                    policy=policy, scenario=self.scenario,
                    batch_size=self.batch_size)

                loss = total_loss.mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                # Evaluate on the test set
                test_total_loss, test_rollout, meta = simulate(
                    policy=policy, scenario=self.scenario, batch_size=31,
                    training=False)
                test_loss = test_total_loss.mean()
                scheduler.step(test_loss)

                losses.append(loss.detach().numpy())
                test_losses.append(test_loss.detach().numpy())

                if test_losses[-1] < best_test_loss:
                    best_test_loss = test_losses[-1]
                    best_model = deepcopy(policy.model)

                pbar.set_description(
                    f"{losses[-1]:1.3f}, {test_losses[-1]:1.3f},"
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

        return test_total_loss, test_rollout, meta


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
        help="torch device"
    )
    a = parser.parse_args()

    # Use the args to construct a full configuration for the experiment.
    config = {
        "name": f"DPC-{a.dr_program}-{a.hidden_dim}",
        "policy_type": "DPC",
        "batch_size": a.batch_size,
        "dr_program": a.dr_program,
        "scenario_config": SCENARIO_TEST if a.dry_run else SCENARIO_DEFAULT,
        "policy_config": {
            "model_config": {
                "hidden_dim": a.hidden_dim,
                "num_intervals": a.num_intervals
            },
            "lr": a.lr,
            "num_epochs": a.num_epochs,
        },
        "training": True,
        "dry_run": a.dry_run
    }
    print("ARGS:", config)

    _ = main(**config)
