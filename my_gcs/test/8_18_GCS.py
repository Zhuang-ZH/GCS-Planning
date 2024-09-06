import cvxpy as cp
import numpy as np
import sys

sys.path.append('../PathPlanning/GCS_Planning/my_gcs')
from graph_of_convex_sets import GraphOfConvexSets, Vertex
import matplotlib.pyplot as plt
from RandomDFS_relaxed import randomForwardPathSearch, findMaxWeightPath
from find_points import linear
from BezierCurves import BezierCurves
from matplotlib.ticker import MultipleLocator

gcs = GraphOfConvexSets()

# 定义regions
regions = [
    {"name": "v1", "center": [-1.75, 0.0], "size": [0.5, 4.0]},
    {"name": "v2", "center": [-1.25, 1.25], "size": [0.5, 1.5]},
    {"name": "v3", "center": [-1.25, -1.75], "size": [0.5, 0.5]},
    {"name": "v4", "center": [-0.75, 1.75], "size": [0.5, 0.5]},
    {"name": "v5", "center": [-0.75, -0.5], "size": [0.5, 3.0]},
    {"name": "v6", "center": [-0.25, 0.0], "size": [0.5, 4.0]},
    {"name": "v7", "center": [0.75, 1.5], "size": [1.5, 1.0]},
    {"name": "v8", "center": [0.5, -1.0], "size": [1.0, 2.0]},
    {"name": "v9", "center": [1.25, -0.25], "size": [0.5, 0.5]},
    {"name": "v10", "center": [1.25, -1.5], "size": [0.5, 1.0]},
    {"name": "v11", "center": [1.75, 0.0], "size": [0.5, 4.0]},
]

# 添加regions到GCS
vertices = gcs.add_region_to_GCS(regions)

# 定义edges
edges = [
    ("v1", "v2"),
    ("v1", "v3"),
    ("v2", "v4"),
    ("v2", "v5"),
    ("v3", "v5"),
    ("v4", "v6"),
    ("v5", "v6"),
    ("v6", "v7"),
    ("v6", "v8"),
    ("v8", "v9"),
    ("v8", "v10"),
    ("v7", "v11"),
    ("v9", "v11"),
    ("v10", "v11"),
]
gcs.add_edge_to_GCS(edges, vertices)

gcs.graphviz()

# 定义起点和终点
start_point = np.array([-1.9, 0])
end_point = np.array([1.9, 0])

start_region, end_region = gcs.find_start_end_region(start_point, end_point, regions)
s = vertices[start_region][0]
t = vertices[end_region][0]

# prob = gcs.solve_shortest_path(s, t)
prob = gcs.solve_shortest_path_relaxation(s, t)
print('Problem status:', prob.status)
print('Optimal value:', prob.value)

# 获取边权重
edge_weights = gcs.get_edge_weights(relaxed=True)

paths = randomForwardPathSearch(regions, edges, edge_weights, start_region, end_region, max_paths=10, max_trials=100, seed=5)
best_path = findMaxWeightPath(paths)
print(best_path)


# 调用线性规划函数
# points = linear(best_path, regions, start_point, end_point)

order = 6
continuity = 1
discrete_points_num = 10

# 调用贝塞尔曲线函数
Bezier_Curves = BezierCurves(best_path, regions, start_point, end_point, order, discrete_points_num, continuity)
# control_points, all_curvatures, curve_points_all = Bezier_Curves.solve()
control_points, all_curvatures, curve_points, all_speed, uniform_points, accerlation, headings = Bezier_Curves.solve()

# gcs.plot_2d()
# gcs.plot_relaxed_subgraph_2d()

# 创建第一个图形窗口
fig1, ax1 = plt.subplots(figsize=(10, 5))
Bezier_Curves.plot_bezier_curve_curvature(ax1, all_curvatures)

# 创建第二个图形窗口
fig2, ax2 = plt.subplots(figsize=(5, 5))
Bezier_Curves.plot_bezier_curve(ax1, curve_points, control_points)

# 显示两个图形窗口
plt.show()