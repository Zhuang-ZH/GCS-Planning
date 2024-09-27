import cvxpy as cp
import numpy as np
import sys
import os

import pandas as pd


# 获取当前路径的父路径
parent_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(parent_path)
# 将父路径添加到 sys.path
sys.path.append(parent_path)
# print(sys.path)

from graph_of_convex_sets import GraphOfConvexSets, Vertex
import matplotlib.pyplot as plt
from RandomDFS_relaxed import randomForwardPathSearch, findMinCostPath
from find_points import linear
from BezierCurves import BezierCurves
from matplotlib.ticker import MultipleLocator
from segment_image import generate_regions, generate_edges, generate_isolated_regions, generate_isolated_edges
from Generate_Track import generate_track_control_points

# 定义地图范围
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# 定义起点和终点
start_point = np.array([-1.9, 0.0])
end_point = np.array([1.9, 0.0])

gcs = GraphOfConvexSets()

obstacle1 = [
    {"name": "ob1", "center": [-1.25, -0.5], "size": [0.5, 2.0]},
    {"name": "ob2", "center": [-0.75, 1.25], "size": [0.5, 0.5]},
    {"name": "ob3", "center": [0.75, 0.5], "size": [1.5, 1.0]},
    {"name": "ob4", "center": [1.25, -0.75], "size": [0.5, 0.5]},
]

obstacle2 = [
    {"name": "ob1", "center": [0.0, 0.0], "size": [1.0, 1.0]},
]

Obstacle = [obstacle1, obstacle2]

num = 2
obstacles = Obstacle[num - 1]

scale_factor = 1.2
expansion_obstacles = []
for obstacle in obstacles:
    expanded_obstacle = {
        "name": obstacle["name"],
        "center": obstacle["center"],
        "size": [dimension * scale_factor for dimension in obstacle["size"]]
    }
    expansion_obstacles.append(expanded_obstacle)

# 生成regions
regions = generate_isolated_regions(x_min, x_max, y_min, y_max, obstacles)

vertices = gcs.add_region_to_GCS(regions)
# print(vertices)

edges = generate_isolated_edges(regions)

gcs.add_edge_to_GCS(edges, vertices)

gcs.graphviz()

start_region, end_region = gcs.find_start_end_region(start_point, end_point, regions)

print(f"start_region: {start_region}")
print(f"end_region: {end_region}")

s = vertices[start_region][0]
t = vertices[end_region][0]

prob = gcs.solve_shortest_path_relaxation(s, t)

print('Problem status:', prob.status)
print('Optimal value:', prob.value)

# 获取边权重
edge_weights = gcs.get_edge_weights(relaxed=True)
# print('edge_weights:',edge_weights)
edge_cost = gcs.get_edge_costs()
# print('edge_cost:',edge_cost)

paths = randomForwardPathSearch(regions, edges, edge_weights, start_region, end_region, max_paths=10, max_trials=100, seed=5)
# print('path:',paths)

best_path = findMinCostPath(paths, edge_cost)
print(best_path)

corridors = [vertices[name][0] for name in best_path]
# print(corridors)

plt.figure()
plt.gca().set_aspect('equal')
plt.axis('on')

gcs.plot_2d(is_plot_edges=False)
# gcs.plot_relaxed_subgraph_2d()
gcs.plot_corridors(corridors)
plt.show()
