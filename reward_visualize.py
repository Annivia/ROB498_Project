import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# Constants and configurations
TARGET_POSE_OBSTACLES = np.array([0.5, 0.0])
OBSTACLE_CENTRE = np.array([0.3, 0.0])
OBSTACLE_RADIUS = 0.1
DISK_RADIUS = 0.05
COLLISION_RANGE = 0.05

with open("config.json", "r") as f:
    config = json.load(f)

def _reward_euclidean(self, state):

    ## Euclidean
    out_penalty = 0
    collision_penalty = 0
    succeed = 0
    if np.any(state < self.observation_space.low) or np.any(state > self.observation_space.high):
        out_penalty = self.config["out_penalty"]
    if self._is_done(state):
        succeed = self.config["succeed_reward"]
    distance_to_target = np.linalg.norm(TARGET_POSE_OBSTACLES - state)
    distance_to_obstacle = np.linalg.norm(OBSTACLE_CENTRE - state)
    
    if distance_to_obstacle < OBSTACLE_RADIUS + DISK_RADIUS + self.config["COLLISION_RANGE"]:
        collision_penalty = self.config["collision_penalty"]

    reward = (10 + collision_penalty + out_penalty + succeed
    - distance_to_target*self.config["distance_scale_factor"])
    
    return reward

def _reward_euclidean_bar(state):

    ## Euclidean Bar
    ## Add marginal reward to help the agent get out of restricted regions
    out_penalty = 0
    collision_penalty = 0
    obstacle_bar = 0
    target_bar = 0
    wall_bar = 0
    BAR_RADIUS = config["BAR_RADIUS"]
    succeed = 0

    if np.linalg.norm(state[:2] - TARGET_POSE_OBSTACLES[:2]) < 1.2 * DISK_RADIUS:
        succeed =  config["succeed_reward"]
    distance_to_target = np.linalg.norm(TARGET_POSE_OBSTACLES - state)
    distance_to_obstacle = np.linalg.norm(OBSTACLE_CENTRE - state)
    distance_to_wall = np.array([
        state[0] - (-0.1),
        state[1] - (-0.35),
        - state[0] + 0.8,
        - state[1] + 0.35
    ]).min()
    
    if distance_to_obstacle < OBSTACLE_RADIUS + DISK_RADIUS + config["COLLISION_RANGE"]:
        collision_penalty = config["collision_penalty"]
    if distance_to_obstacle < OBSTACLE_RADIUS + DISK_RADIUS + BAR_RADIUS:
        obstacle_bar = - OBSTACLE_RADIUS - BAR_RADIUS - DISK_RADIUS + distance_to_obstacle
    if distance_to_target < BAR_RADIUS + 1.2*DISK_RADIUS:
        target_bar = BAR_RADIUS + 1.2*DISK_RADIUS - distance_to_target
    if distance_to_wall < BAR_RADIUS and distance_to_wall > 0:
        wall_bar = distance_to_wall - BAR_RADIUS

    reward = (10 + succeed + collision_penalty + out_penalty 
    - distance_to_target*config["distance_scale_factor"] 
    + obstacle_bar*config["obstacle_bar_factor"] 
    + target_bar*config["target_bar_factor"]
    + wall_bar*config["wall_bar_factor"])

    return reward

def _reward_square(self, state):

    ## Square
    out_penalty = 0
    collision_penalty = 0
    succeed = 0
    if np.any(state < self.observation_space.low) or np.any(state > self.observation_space.high):
        out_penalty = self.config["out_penalty"]
    if self._is_done(state):
        succeed = self.config["succeed_reward"]
    distance_to_target = np.linalg.norm(TARGET_POSE_OBSTACLES - state)
    distance_to_obstacle = np.linalg.norm(OBSTACLE_CENTRE - state)
    
    if distance_to_obstacle < OBSTACLE_RADIUS + DISK_RADIUS + self.config["COLLISION_RANGE"]:
        collision_penalty = self.config["collision_penalty"]

    reward = (collision_penalty + out_penalty + succeed
    - np.log(distance_to_target)*self.config["distance_scale_factor"])

    return reward

def _reward_square_bar(self, state):

    ## Square Bar
    ## Add marginal reward to help the agent get out of restricted regions
    out_penalty = 0
    collision_penalty = 0
    obstacle_bar = 0
    target_bar = 0
    wall_bar = 0
    BAR_RADIUS = self.config["BAR_RADIUS"]
    succeed = 0

    if np.any(state < self.observation_space.low) or np.any(state > self.observation_space.high):
        out_penalty = self.config["out_penalty"]
    if self._is_done(state):
        succeed = self.config["succeed_reward"]
    distance_to_target = np.linalg.norm(TARGET_POSE_OBSTACLES - state)
    distance_to_obstacle = np.linalg.norm(OBSTACLE_CENTRE - state)
    distance_to_wall = np.array([
        state[0] - self.observation_space.low[0],
        state[1] - self.observation_space.low[1],
        - state[0] + self.observation_space.high[0],
        - state[1] + self.observation_space.high[1]
    ]).min()

    if distance_to_obstacle < OBSTACLE_RADIUS + DISK_RADIUS + self.config["COLLISION_RANGE"]:
        collision_penalty = self.config["collision_penalty"]
    if distance_to_obstacle < OBSTACLE_RADIUS + DISK_RADIUS + BAR_RADIUS:
        obstacle_bar = - OBSTACLE_RADIUS - BAR_RADIUS - DISK_RADIUS + distance_to_obstacle
    if distance_to_target < BAR_RADIUS + 1.2*DISK_RADIUS:
        target_bar = BAR_RADIUS + 1.2*DISK_RADIUS - distance_to_target
    if distance_to_wall < BAR_RADIUS and distance_to_wall > 0:
        wall_bar = distance_to_wall - BAR_RADIUS

    reward = (collision_penalty + out_penalty + succeed 
    - np.log(distance_to_target)*self.config["distance_scale_factor"] 
    + obstacle_bar*self.config["obstacle_bar_factor"] 
    + target_bar*self.config["target_bar_factor"]
    + wall_bar*self.config["wall_bar_factor"])

    return reward


def visualize_reward_function(reward_function):

    x = np.linspace(-0.1, 0.8, 100)
    y = np.linspace(-0.35, 0.35, 100)
    X, Y = np.meshgrid(x, y)

    reward_matrix = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            reward_matrix[i, j] = reward_function(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, reward_matrix, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Reward")
    if (reward_function == _reward_euclidean):
        ax.set_title("Eucildean Reward Function")
    elif (reward_function == _reward_euclidean_bar):
        ax.set_title("Eucildean Bar Reward Function")
    elif (reward_function == _reward_square):
        ax.set_title("Logarithm Reward Function")
    elif (reward_function == _reward_square_bar):
        ax.set_title("Logarithm Bar Reward Function")
    plt.show()

# Call the visualize_reward_function with the desired reward function
visualize_reward_function(_reward_euclidean_bar)  # Change this to the desired reward function

