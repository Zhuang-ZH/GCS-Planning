"""
Plotting tools for Sampling-based algorithms
@author: huiming zhou
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

import env


class Plotting:
    def __init__(self, x_start, x_goal, ax):
        self.xI, self.xG = x_start, x_goal
        self.env = env.Env()
        self.obs_bound = self.env.obs_boundary
        self.obs_rectangle = self.env.obs_rectangle
        self.ax = ax

    def plot_grid(self, name):
        for (ox, oy, w, h) in self.obs_bound:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        self.ax.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        self.ax.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        self.ax.set_title(name)
        self.ax.axis("equal")

# def main():
#     x_start = (-1.9, 0)  # Starting node
#     x_goal = (1.9, 0)  # Goal node

#     obstacle = Plotting(x_start, x_goal)
#     obstacle.plot_grid("obstacle")

# if __name__ == '__main__':
#     main()
