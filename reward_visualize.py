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


def _reward(state):

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

# Create a mesh grid for x and y state spaces
x = np.linspace(-0.1, 0.8, 100)
y = np.linspace(-0.35, 0.35, 100)
X, Y = np.meshgrid(x, y)

# Calculate the reward for each point in the mesh grid
reward_matrix = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        reward_matrix[i, j] = _reward(np.array([X[i, j], Y[i, j]]))

# Create a 3D plot of the reward function
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, reward_matrix, cmap="viridis")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Reward")
ax.set_title("Reward Function")
plt.show()
