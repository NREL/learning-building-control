import os

import ray
import numpy as np

from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from lbc.building_env import BuildingControlEnv
from lbc.demand_response import DemandResponseProgram as DRP
from lbc.scenario import Scenario

from custom_model import get_customized_model

tf1, tf, tfv = try_import_tf()


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
    parser.add_argument('--num_lookahead_steps', type=int, default=24)
    parser.add_argument('--train_batch_size', type=int, default=5000)
    parser.add_argument('--episodes_per_batch', type=int, default=5000)
    parser.add_argument('--dr_program', type=str, default='RTP')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--run_hour', type=float, default=0.9)
    args = parser.parse_args()

    dr_program_type = args.dr_program
    env_name = ('BuildingControl' + dr_program_type
                + str(args.num_lookahead_steps) + 'Env-v0')

    if args.redis_password is None:
        # Single HPC node
        ray.init(_temp_dir="/tmp/scratch/ray")
        # # Local machine
        # ray.init(_node_ip_address='192.168.0.16')
    else:
        # On an HPC cluster
        ray.init(_redis_password=args.redis_password,
                 address=args.ip_head)

    def env_creator(config):
        drp = DRP(dr_program_type)
        s = Scenario(dr_program=drp)
        env = BuildingControlEnv(scenario=s,
                                 num_lookahead_steps=args.num_lookahead_steps)
        return env

    register_env(env_name, env_creator)
    obs_dim = env_creator(None).observation_space.shape[0]
    custom_model = get_customized_model(obs_dim, [128, 128],
                                        tf.nn.relu, tf.sigmoid)
    ModelCatalog.register_custom_model("customized_model", custom_model)

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
            # Use the default network model
            "fcnet_hiddens": [128, 128],
            # # Use a customized network when using PPO algorithm.
            # "custom_model": "customized_model"
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
