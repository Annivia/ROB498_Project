import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import argparse
from helper import *
import os

def translate(reward):
    if reward == 'euclidean':
        return 'Euclidean'
    elif reward == 'euclidean_bar':
        return 'Euclidean Bar'
    elif reward == 'square':
        return 'Logarithm'
    elif reward == 'square_bar':
        return 'Logarithm Bar'

def draw_trajectory(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)

    data.sort(key=lambda x: x["checkpoint"])

    reward = extract_reward(file_name)
    algo = extract_algorithm(file_name)
    save_name = os.path.join('figures', algo+'_'+reward+'.png')

    with open("config.json", "r") as f:
        config = json.load(f)

    TARGET_POSE_OBSTACLES = config["TARGET_POSE_OBSTACLES"]
    OBSTACLE_RADIUS = config["OBSTACLE_RADIUS"]
    OBSTACLE_CENTRE = config["OBSTACLE_CENTRE"]
    DISK_RADIUS = config["DISK_RADIUS"]
    BAR_RADIUS = config["BAR_RADIUS"]
    COLLISION_RANGE = config["COLLISION_RANGE"]

    fig, ax = plt.subplots()
    cmap = cm.get_cmap("viridis", len(data))

    for idx, entry in enumerate(data):
        checkpoint = entry["checkpoint"]
        states = entry["state"]
        x_coords, y_coords = zip(*states)
        ax.plot(x_coords, y_coords, color=cmap(idx), label=f"Checkpoint {checkpoint}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0.1, 0.8)
    ax.set_ylim(-0.35, 0.35)

    if reward == 'euclidean_bar' or reward == 'square_bar':
        circlebar1 = patches.Circle((OBSTACLE_CENTRE[0], OBSTACLE_CENTRE[1]), radius=OBSTACLE_RADIUS+DISK_RADIUS+BAR_RADIUS, facecolor='white', edgecolor='black', alpha=0.3)
        circlebar2 = patches.Circle((TARGET_POSE_OBSTACLES[0], TARGET_POSE_OBSTACLES[1]), radius=1.2*DISK_RADIUS+BAR_RADIUS, facecolor='white', edgecolor='black', alpha=0.3)
        ax.add_patch(circlebar1)
        ax.add_patch(circlebar2)

        x1 = 0.1 + BAR_RADIUS
        x2 = 0.8 - BAR_RADIUS
        y1 = -0.35 + BAR_RADIUS
        y2 = 0.35 - BAR_RADIUS

        ax.plot([x1, x1], [y1, y2], 'gray', alpha=0.3)
        ax.plot([x2, x2], [y1, y2], 'gray', alpha=0.3)
        ax.plot([x1, x2], [y1, y1], 'gray', alpha=0.3)
        ax.plot([x1, x2], [y2, y2], 'gray', alpha=0.3)


    circle0 = patches.Circle((OBSTACLE_CENTRE[0], OBSTACLE_CENTRE[1]), radius=OBSTACLE_RADIUS+DISK_RADIUS, facecolor='grey', edgecolor='black', alpha=0.6)
    circle1 = patches.Circle((OBSTACLE_CENTRE[0], OBSTACLE_CENTRE[1]), radius=OBSTACLE_RADIUS+DISK_RADIUS+COLLISION_RANGE, facecolor='black', edgecolor='black', alpha=0.4)
    circle2 = patches.Circle((TARGET_POSE_OBSTACLES[0], TARGET_POSE_OBSTACLES[1]), radius=1.2*DISK_RADIUS, facecolor='green', edgecolor='black', alpha=0.6)
    ax.add_patch(circle0)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_aspect('equal')
    ax.legend(fontsize='small', loc='center left')
    
    plt.title('Trajectory of Disk Center with ' + algo + ', ' + translate(reward))
    plt.savefig(save_name, dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    draw_trajectory(args.dir)

