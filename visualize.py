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


if __name__ == "__main__":
    
    checkpoint_dir='/home/pony/Documents/Umich/ROB498/Final_Project/codes/checkpoints/PPO_square_bar_04-19_06-24/checkpoint_000007'
    reward_type = extract_reward(checkpoint_dir)
    algorithm = extract_algorithm(checkpoint_dir)

    if (reward_type == 'euclidean'):
        register_env("PandaDiskPushingEnv", create_environment_euclidean)
        env = PandaDiskPushingEnv_euclidean(visualizer=True, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
    elif (reward_type == 'euclidean_bar'):
        register_env("PandaDiskPushingEnv", create_environment_euclidean_bar)
        env = PandaDiskPushingEnv_euclidean_bar(visualizer=True, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
    elif (reward_type == 'square'):
        register_env("PandaDiskPushingEnv", create_environment_square)
        env = PandaDiskPushingEnv_square(visualizer=True, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
    elif (reward_type == 'square_bar'):
        register_env("PandaDiskPushingEnv", create_environment_square_bar)
        env = PandaDiskPushingEnv_square_bar(visualizer=True, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
    else:
        print("Invalid Checkpoint Directory")
        exit(1)

    agent = set_up(algorithm)
    agent.restore(checkpoint_dir)

    state = env.reset()
    sum_reward = 0
    n_step = 40

    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        print("State: ", state, " | Reward: ", reward)
        sum_reward += reward

        # env.render()

        if done == 1:
            # report at the end of each episode
            # print("cumulative reward", sum_reward)
            print("Goal Reached with reward = ", reward)
            sum_reward = 0
            break
