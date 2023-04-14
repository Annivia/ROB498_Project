from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig
from ray.rllib.algorithms.ddpg.ddpg import DDPG, DDPGConfig
from ray.rllib.algorithms.sac.sac import SAC, SACConfig
import shutil
from panda_pushing_env import PandaDiskPushingEnv
import argparse
import sys
import datetime


def create_environment(env_config):
    return PandaDiskPushingEnv(**env_config)

def set_up(algorithm):

    if algorithm == "PPO":
        config = PPOConfig()#.environment(PandaDiskPushingEnv, env_config=env_config)
        config = config.resources(num_gpus=1)  
        config = config.rollouts(num_rollout_workers=2) 
        config = config.framework('torch')
        agent = PPO(config, env=PandaDiskPushingEnv)

    # if algorithm == "PPO":
    #     config = ppo.DEFAULT_CONFIG.copy()
    #     agent = ppo.PPOTrainer(config, env=PandaDiskPushingEnv)
    
    # elif algorithm == "DQN":
    #     config = DQNConfig().copy()
    #     config["log_level"] = "WARN"
    #     agent = DQN(config, env=PandaDiskPushingEnv)

    elif algorithm == "DDPG":
        config = DDPGConfig()
        config = config.resources(num_gpus=1)  
        config = config.rollouts(num_rollout_workers=2) 
        config = config.framework('torch')
        agent = DDPG(config, env=PandaDiskPushingEnv)

    elif algorithm == "SAC":
        config = SACConfig().training(gamma=0.9, lr=0.01, tau=0.005)
        config = config.resources(num_gpus=1)  
        config = config.rollouts(num_rollout_workers=2) 
        config = config.framework('torch')
        agent = SAC(config, env=PandaDiskPushingEnv)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return agent


def train(algorithm, iter_num):

    # init directory in which to save checkpoints
    now = datetime.datetime.now()
    date_time_str = now.strftime("%m-%d_%H-%M")
    chkpt_root = "checkpoints/" + algorithm + '_' + date_time_str

    # # Create the directory
    # os.makedirs(dir_name)
    # chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    register_env("PandaDiskPushingEnv", create_environment)

    # configure the environment and create agent
    config, agent = set_up(algorithm)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

    # train a policy with RLlib using PPO
    for n in range(iter_num):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    # print(model.base_model.summary())


def test(algorithm, checkpoint_dir):

    config, agent = set_up(algorithm)
    chkpt_file='tmp/exa/checkpoint_000005'
    # chkpt_file=checkpoint_dir

    agent.restore(chkpt_file)
    env = PandaDiskPushingEnv(visualizer=None, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=1)

    state = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        print(state)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--iter_num", type=int, default=15)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str)

    args = parser.parse_args()
    print("Algorithm: ", args.algorithm)

    if args.train and args.test:
        print("Error: Cannot run in both train and test modes. Please specify only one mode.")
        sys.exit(1)

    if args.train:
        print("Running in train mode...")
        train(args.algorithm, args.iter_num)

    elif args.test:
        print("Running in test mode...")
        if args.checkpoint_dir is None:
            print("Error: Please provide a checkpoint directory with --checkpoint_dir when running in test mode.")
            sys.exit(1)
        test(args.algorithm, args.checkpoint_dir)

    else:
        print("Error: Please specify either train or test mode.")
        sys.exit(1)
