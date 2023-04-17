import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import argparse
from helper import *
import os

def draw_trajectory(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)

    data.sort(key=lambda x: x["checkpoint"])

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

    circle1 = patches.Circle((0.575, -0.05), radius=0.05, facecolor='black', edgecolor='black')
    circle2 = patches.Circle((0.75, -0.1), radius=0.06, facecolor='yellow', edgecolor='black')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_aspect('equal')
    ax.legend()

    reward = extract_reward(file_name)
    algo = extract_algorithm(file_name)
    save_name = os.path.join('..', 'figures', algo+'_'+reward+'.png')

    plt.savefig(save_name, dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    draw_trajectory(args.dir)

