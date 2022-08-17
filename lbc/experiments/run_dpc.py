from copy import deepcopy
import logging
import time

from tqdm import tqdm

import torch

from lbc.experiments.runner import PolicyRunner, save_runner
from lbc.simulate import simulate


logger = logging.getLogger(__file__)


class DPCRunner(PolicyRunner):

    @property
    def name(self):
        lr = self.policy_config["lr"]
        la = self.policy.num_lookahead_steps
        return f"DPC-{self.dr_program}-{la}-{lr:0.3f}" + self.name_ext

    def train_policy(self, **kwargs):

        policy = self.policy

        opt = torch.optim.Adam(
            policy.model.parameters(), lr=self.policy_config["lr"])

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt, factor=0.5, patience=10, cooldown=10)
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     opt, step_size=500, gamma=np.sqrt(.1))

        losses = []
        test_losses = []
        tic = time.time()
        pbar = tqdm(range(self.policy_config["num_epochs"]))
        for _ in pbar:
            try:
                # Simulate on the training set and perform gradient update
                loss, rollout, meta = simulate(
                    policy=policy, scenario=self.scenario,
                    batch_size=self.batch_size, training=True)

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
                    policy=policy, scenario=self.scenario,
                    batch_size=self.batch_size, training=False)
                test_loss = test_loss.mean()

                # Track the training and test losses.
                losses.append(loss.detach().numpy())
                test_losses.append(test_loss.detach().numpy())

                imitation_cost = rollout.data["imitation_cost"].sum(-1).mean()

                pbar.set_description(
                    f"{losses[-1]:1.3f}, {test_losses[-1]:1.3f}"
                    f"{imitation_cost:1.3f}")

            except KeyboardInterrupt:
                logger.info("stopped")
                break

        meta.update({
            "model": deepcopy(policy.model),
            "losses": losses,
            "test_losses": test_losses,
            "train_time": time.time() - tic
        })

        return loss, rollout, meta

    def run_policy(self, batch_size=None, training=False):

        batch_size = batch_size if batch_size is not None else self.batch_size

        loss, rollout, meta = simulate(
            policy=self.policy, scenario=self.scenario, batch_size=batch_size,
            training=training)

        return loss, rollout, meta


def main(**config):

    runner = DPCRunner(**config)

    train_data = runner.train_policy()
    test_data = runner.run()

    return save_runner(runner=runner, config=config,
                       test_data=test_data, train_data=train_data)


if __name__ == "__main__":

    from lbc.experiments.runner import get_parser
    from lbc.experiments.config import get_config

    parser = get_parser()
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="number of training epochs"
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=24,
        help="number of lookahead steps"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
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

    config = get_config("DPC", **vars(a))
    config["policy_config"] = {
        "model_config": {
            "hidden_dim": a.hidden_dim,
        },
        "num_lookahead_steps": a.lookahead,
        "lr": a.lr,
        "num_epochs": a.num_epochs,
        "device": a.device
    }
    logger.info(f"CONFIG: {config}")

    _ = main(**config)
