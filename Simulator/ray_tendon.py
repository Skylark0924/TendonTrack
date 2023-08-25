import argparse
import cv2

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

# from env.gym_tendon import TendonGymEnv
from env.gym_tracking_tendon import TendonGymEnv


def config_env(args):
    """preparing config for environment"""
    config = None
    return config


def config_agent(args, env_config=None):
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = args.gpus
    config["num_workers"] = args.workers
    config["timesteps_per_iteration"] = args.num_step
    config["env"] = "TendonGym-v0"
    # config["env_config"] = env_config
    return config


def main(args):
    ray.init()
    # build UR5 environment
    register_env("TendonGym-v0", lambda config: TendonGymEnv())
    # env_config = config_env(args)
    agent_config = config_agent(args)

    if args.tune is False:
        trainer = PPOTrainer(
            env="TendonGym-v0",
            config=agent_config)
        for i in range(args.episode):
            # Perform one iteration of training the policy with PPO
            result = trainer.train()
            print(pretty_print(result))

            if i % 30 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
    else:
        tune.run(
            args.algo,
            stop={"training_iteration": 10},
            config={
                "env": "TendonGym-v0",
                "num_workers": args.workers,
                "timesteps_per_iteration": 500,
                "train_batch_size": 256,
                "sample_batch_size": 256
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='SAC', choices=['PPO', 'SAC'],
                        help='choose an algorithm')
    parser.add_argument('--ckpt', type=str, help='inference or training')
    parser.add_argument('--episode', type=int, default=10, help='number of training episodes')
    parser.add_argument('--num_step', type=int, default=10 ** 3,
                        help='number of timesteps for one episode, and for inference')
    parser.add_argument('--batch_size', type=int, default=128, help='model saving frequency')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--tune', dest='tune', action='store_true')
    parser.add_argument('--no-tune', dest='tune', action='store_false')
    parser.set_defaults(tune=True)
    parser.add_argument('--eager', dest='eager', action='store_true')
    parser.add_argument('--no-eager', dest='eager', action='store_false')
    parser.set_defaults(eager=False)
    args = parser.parse_args()

    main(args)
