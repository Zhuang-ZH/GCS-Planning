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
    {"name": "v1", "center": [2.5, 2.0], "size": [5.0, 4.0]},
    {"name": "v2", "center": [6.0, 3.0], "size": [2.0, 2.0]},
    {"name": "v3", "center": [6.0, 0.5], "size": [2.0, 1.0]},
    {"name": "v4", "center": [8.0, 2.0], "size": [2.0, 4.0]},
    {"name": "v5", "center": [10.0, 3.5], "size": [2.0,1.0]},
    {"name": "v6", "center": [10.0, 1.0], "size": [2.0, 2.0]},
    {"name": "v7", "center": [12.0, 2.0], "size": [2.0, 4.0]},
    {"name": "v8", "center": [14.0, 3.0], "size": [2.0, 2.0]},
    {"name": "v9", "center": [14.0, 0.5], "size": [2.0, 1.0]},
    {"name": "v10", "center": [15.5, 2.0], "size": [1.0, 4.0]},
    {"name": "v11", "center": [17.0, 1.5], "size": [2.0, 3.0]},
    {"name": "v12", "center": [19.0, 2.0], "size": [2.0, 4.0]},
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
    ("v1", "v3"),
    ("v2", "v4"),
    ("v3", "v4"),
    ("v4", "v5"),
    ("v4", "v6"),
    ("v5", "v7"),
    ("v6", "v7"),
    ("v7", "v8"),
    ("v7", "v9"),
    ("v8", "v10"),
    ("v9", "v10"),
    ("v10", "v11"),
    ("v11", "v12"),
]

# 添加edges到GCS
for (start, end) in edges:
    v_start, x_start = vertices[start]
    v_end, x_end = vertices[end]
    edge = gcs.add_edge(v_start, v_end)
    edge.add_cost(cp.norm(x_start - x_end, 2))
    edge.add_constraint(x_end[1] >= x_start[1])

gcs.graphviz()

start_point = np.array([1.0, 1.5])
end_point = np.array([19.0, 1.5])

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
border_rect = plt.Rectangle((0, 0), 20, 4, edgecolor='black', facecolor='none', linewidth=2)
ax.add_patch(border_rect)

ax.plot(start_point[0], start_point[1], 'go')  # 起点绿色圆点
ax.plot(end_point[0], end_point[1], 'ro')  # 终点红色圆点

# 设置图形属性
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Path Visualization')

plt.show()