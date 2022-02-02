import os

import ray
import numpy as np

from ray import tune
from ray.tune.registry import register_env

from lbc.building_env import BuildingControlEnv


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# LOG_PATH = os.path.join(parent_dir, 'results/')  # LOCAL
LOG_PATH = os.path.join('/scratch/xzhang2/', 'lbc/')  # HPC


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='PPO')
    parser.add_argument('--redis_password', type=str, default=None)
    parser.add_argument('--ip_head', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=5000)
    parser.add_argument('--episodes_per_batch', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--run_hour', type=float, default=0.9)
    args = parser.parse_args()

    env_name = 'BuildingControlTOUEnv-v0'

    if args.redis_password is None:
        # Single node
        ray.init(_temp_dir="/tmp/scratch/ray")
        # ray.init(_node_ip_address='192.168.0.36')
    else:
        # On a cluster
        ray.init(_redis_password=args.redis_password,
                 address=args.ip_head)

    def env_creater(config):
        env = BuildingControlEnv()
        return env

    register_env(env_name, env_creater)

    algo_specific_config = {
        'ES': {
            "stepsize": args.lr,
            "episodes_per_batch": args.episodes_per_batch,
            "noise_stdev": args.sigma,
            "eval_prob": 0.03,
        },
        'PPO': {
            "train_batch_size": args.train_batch_size,
        }
    }

    config = {
        "env": env_name,
        "model": {
            "fcnet_hiddens": [256, 256],
        },
        "num_workers": args.num_workers,
    }
    config.update(algo_specific_config[args.algo])

    def trial_name_id(trial):
        rndstr = str(np.random.randint(500))
        return f"{trial.trainable_name}_{env_name}_{trial.trial_id}_{rndstr}"

    run_time = 3600 * args.run_hour \
        if args.run_hour < 1.0 else 3600 * args.run_hour - 300

    tune.run(
        args.algo,
        name=env_name,
        stop={
            # "training_iteration": 10,
            "time_total_s": run_time
        },
        checkpoint_freq=5,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=3,
        local_dir=LOG_PATH,
        config=config,
        trial_name_creator=trial_name_id,
    )

    ray.shutdown()


if __name__ == '__main__':
    main()
