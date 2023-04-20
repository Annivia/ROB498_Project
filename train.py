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
from panda_pushing_env_euclidean import PandaDiskPushingEnv_euclidean
from panda_pushing_env_square import PandaDiskPushingEnv_square
from panda_pushing_env_euclidean_bar import PandaDiskPushingEnv_euclidean_bar
from panda_pushing_env_square_bar import PandaDiskPushingEnv_square_bar
import argparse
import sys
import datetime
from helper import *
import pandas as pd
import json


def create_environment_euclidean(env_config):
    return PandaDiskPushingEnv_euclidean(**env_config)

def create_environment_euclidean_bar(env_config):
    return PandaDiskPushingEnv_euclidean_bar(**env_config)

def create_environment_square(env_config):
    return PandaDiskPushingEnv_square(**env_config)

def create_environment_square_bar(env_config):
    return PandaDiskPushingEnv_square_bar(**env_config)


def set_up(algorithm):

    if algorithm == "PPO":
        config = PPOConfig()#.environment(PandaDiskPushingEnv, env_config=env_config)
        config = config.resources(num_gpus=1)  
        config = config.rollouts(num_rollout_workers=2) 
        config = config.framework('torch')
        agent = PPO(config, env='PandaDiskPushingEnv')

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
        agent = DDPG(config, env='PandaDiskPushingEnv')

    elif algorithm == "SAC":
        config = SACConfig().training(gamma=0.9, lr=0.01, tau=0.005)
        config = config.resources(num_gpus=1)
        config = config.rollouts(num_rollout_workers=2)
        config = config.framework('torch')
        agent = SAC(config, env='PandaDiskPushingEnv')

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return agent


def train(algorithm, iter_num, reward):

    # init directory in which to save checkpoints
    now = datetime.datetime.now()
    date_time_str = now.strftime("%m-%d_%H-%M")
    chkpt_root = "checkpoints/" + algorithm + '_' + reward + "_" + date_time_str

    # # Create the directory
    # os.makedirs(chkpt_root)
    # chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    if reward == 'euclidean':
        register_env("PandaDiskPushingEnv", create_environment_euclidean)
    elif reward == 'square':
        register_env("PandaDiskPushingEnv", create_environment_square)
    elif reward == 'euclidean_bar':
        register_env("PandaDiskPushingEnv", create_environment_euclidean_bar)
    elif reward == 'square_bar':
        register_env("PandaDiskPushingEnv", create_environment_square_bar)

    # configure the environment and create agent
    agent = set_up(algorithm)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

    results_df = pd.DataFrame(columns=["Iteration", "Reward Min", "Reward Mean", "Reward Max", "Episode Length Mean"])

    # train a policy with RLlib using PPO
    for n in range(iter_num):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        # print(status.format(
        #         n + 1,
        #         result["episode_reward_min"],
        #         result["episode_reward_mean"],
        #         result["episode_reward_max"],
        #         result["episode_len_mean"],
        #         chkpt_file
        #         ))

        # Add results to DataFrame
        results_df = results_df.append({
            "Iteration": n + 1,
            "Reward Min": result["episode_reward_min"],
            "Reward Mean": result["episode_reward_mean"],
            "Reward Max": result["episode_reward_max"],
            "Episode Length Mean": result["episode_len_mean"]
        }, ignore_index=True)

    # Save DataFrame as CSV
    csv_file_path = os.path.join(chkpt_root, 'train_results.csv')
    results_df.to_csv(csv_file_path, index=False)


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    # print(model.base_model.summary())


def test(checkpoint_dir):
    
    chkpt_file=checkpoint_dir
    reward_type = extract_reward(checkpoint_dir)
    algorithm = extract_algorithm(checkpoint_dir)
    state_log = []
    reward_log = []

    if (reward_type == 'euclidean'):
        register_env("PandaDiskPushingEnv", create_environment_euclidean)
        env = PandaDiskPushingEnv_euclidean(visualizer=None, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=1)
    elif (reward_type == 'euclidean_bar'):
        register_env("PandaDiskPushingEnv", create_environment_euclidean_bar)
        env = PandaDiskPushingEnv_euclidean_bar(visualizer=None, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=1)
    elif (reward_type == 'square'):
        register_env("PandaDiskPushingEnv", create_environment_square)
        env = PandaDiskPushingEnv_square(visualizer=None, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=1)
    elif (reward_type == 'square_bar'):
        register_env("PandaDiskPushingEnv", create_environment_square_bar)
        env = PandaDiskPushingEnv_square_bar(visualizer=None, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=1)
    else:
        print("Invalid Checkpoint Directory")
        exit(1)

    agent = set_up(algorithm)
    agent.restore(chkpt_file)

    state = env.reset()
    sum_reward = 0
    n_step = 40

    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        state_log.append(state)
        reward_log.append(reward)
        # print("State: ", state, " | Reward: ", reward)
        sum_reward += reward

        # env.render()

        if done == 1:
            # report at the end of each episode
            # print("cumulative reward", sum_reward)
            # print("Goal Reached with reward = ", reward)
            state = env.reset()
            sum_reward = 0
            break

    return state_log, reward_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--iter_num", type=int, default=15)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--reward", type=str)

    args = parser.parse_args()
    print("Algorithm: ", args.algorithm)

    if args.train and args.test:
        print("Error: Cannot run in both train and test modes. Please specify only one mode.")
        sys.exit(1)

    if args.train:
        print("Running in train mode...")
        train(args.algorithm, args.iter_num, args.reward)

    elif args.test:
        print("Running in test mode...")
        if args.checkpoint_dir is None:
            print("Error: Please provide a checkpoint directory with --checkpoint_dir when running in test mode.")
            sys.exit(1)

        results = []

        for _ in range(1):
            states, rewards = test(args.checkpoint_dir)
            checkpoint_number = extract_checkpoint_id(args.checkpoint_dir)

            if checkpoint_number is not None:
                states_list = [state.tolist() for state in states]
                rewards_list = rewards
                result = {"checkpoint": checkpoint_number, "state": states_list, "reward": rewards_list}
                results.append(result)

        # Save the results in a JSON file in the parent folder of checkpoint_dir
        parent_folder = os.path.dirname(os.path.abspath(args.checkpoint_dir))
        results_file = os.path.join(parent_folder, "results.json")
        # print("Writing into ", results_file)

        # with open(results_file, "w") as f:
        #     json.dump(results, f)

        # Check if the JSON file exists
        if os.path.isfile(results_file):
            # If it exists, read the content into a list
            with open(results_file, "r") as f:
                existing_results = json.load(f)
        else:
            # If it doesn't exist, initialize an empty list
            existing_results = []

        # Append the new results to the existing results
        existing_results.extend(results)

        # Write the updated content back to the JSON file
        with open(results_file, "w") as f:
            json.dump(existing_results, f)


    else:
        print("Error: Please specify either train or test mode.")
        sys.exit(1)
