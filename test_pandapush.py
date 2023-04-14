import os
import numpy as np
import gym
from gym import spaces
import pybullet as p
import pybullet_data as pd
from ray import tune
from ray.tune.registry import register_env
from panda_pushing_env import PandaDiskPushingEnv


# Ensure your PandaDiskPushingEnv class is defined before this point

def create_environment(env_config):
    return PandaDiskPushingEnv(**env_config)

register_env("PandaDiskPushingEnv", create_environment)
# env = gym.make("PandaDiskPushingEnv")
# print("Observation space:", env.observation_space)

env = PandaDiskPushingEnv(render_non_push_motions=False, camera_heigh=500, camera_width=500, render_every_n_steps=5)
env.reset()
print("Observation space:", env.observation_space)

config = {
    "env": "PandaDiskPushingEnv",
    "env_config": {
        "debug": False,
        "include_obstacle": True,
        "render_every_n_steps": 1,
        "done_at_goal": True,
        "camera_heigh": 800,
        "camera_width": 800,
    },
    "num_gpus": 0,
    "num_workers": 1,
    "lr": 3e-4,
    "framework": "torch",
    "monitor": True,
}

tune.run("PPO", config=config, stop={"timesteps_total": 1000000}, checkpoint_freq=10, checkpoint_at_end=True)
