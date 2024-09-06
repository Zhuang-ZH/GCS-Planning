import cvxpy as cp
import numpy as np
import sys

sys.path.append('../PathPlanning/GCS_Planning/my_gcs')
from graph_of_convex_sets import GraphOfConvexSets, Vertex
import matplotlib.pyplot as plt
from RandomDFS_relaxed import randomForwardPathSearch, findMaxWeightPath
from find_points import linear
from BezierCurves import BezierCurves

gcs = GraphOfConvexSets()

# 定义regions
regions = [
    {"name": "v1", "center": [1.5, 0.75], "size": [3.0, 2.5]},
    {"name": "v2", "center": [4.0, 1.25], "size": [2.0, 1.5]},
    {"name": "v3", "center": [5.5, 0.75], "size": [1.0, 2.5]},
]

# 添加regions到GCS
vertices = {}
for region in regions:
    v = gcs.add_vertex(region["name"])
    x = v.add_variable(2)
    c = np.array(region["center"])
    size = np.array(region["size"]) / 2
    # 添加长方形约束
    constraint = cp.abs(x - c) <= size
    v.add_constraint(constraint)
    vertices[region["name"]] = (v, x)

# 定义edges
edges = [
    ("v1", "v2"),
    ("v2", "v3")
]

# 添加edges到GCS
for (start, end) in edges:
    v_start, x_start = vertices[start]
    v_end, x_end = vertices[end]
    edge = gcs.add_edge(v_start, v_end)
    edge.add_cost(cp.norm(x_start - x_end, 2))
    edge.add_constraint(x_end[1] >= x_start[1])

gcs.graphviz()

start_point = np.array([0.0, -0.5])
end_point = np.array([6.0, 1.0])

# 绘制区域和路径
fig, ax = plt.subplots()

# gcs.plot_2d()
gcs.plot_relaxed_subgraph_2d()

# 绘制每个区域
for region in regions:
    center = np.array(region["center"])
    size = np.array(region["size"])
    lower_left = center - size / 2
    rect = plt.Rectangle(lower_left, size[0], size[1], edgecolor='black', facecolor='lightgray')
    ax.add_patch(rect)
    ax.text(center[0], center[1], region["name"], ha='center', va='center', fontsize=8, color='black')

# 绘制地图边框
border_rect = plt.Rectangle((0.0, -0.5), 6.0, 2.5, edgecolor='black', facecolor='none', linewidth=2)
ax.add_patch(border_rect)

ax.plot(start_point[0], start_point[1], 'go')  # 起点绿色圆点
ax.plot(end_point[0], end_point[1], 'ro')  # 终点红色圆点

# 设置图形属性
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Path Visualization')

plt.show()