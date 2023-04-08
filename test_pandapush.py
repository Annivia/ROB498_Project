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

config = {
    "env": "PandaDiskPushingEnv",
    "env_config": {
        "debug": False,
        "include_obstacle": True,
        "render_every_n_steps": 1,
        "done_at_goal": True,
    },
    "num_gpus": 0,
    "num_workers": 1,
    "lr": 3e-4,
    "framework": "torch",
}

tune.run("PPO", config=config, stop={"timesteps_total": 1000000}, checkpoint_freq=10, checkpoint_at_end=True)
