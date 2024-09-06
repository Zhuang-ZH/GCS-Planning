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

regions = [
    {"name": "v1", "center": [1.5, 0.75], "size": [3.0, 2.5]},
    {"name": "v2", "center": [4.0, 1.25], "size": [2.0, 1.5]},
    {"name": "v3", "center": [5.5, 0.75], "size": [1.0, 2.5]},
]

vertices = gcs.add_region_to_GCS(regions)

edges = [
    ("v1", "v2"),
    ("v2", "v3")
]
gcs.add_edge_to_GCS(edges, vertices)

gcs.graphviz()

start_point = np.array([0.0, -0.5])
end_point = np.array([6.0, 1.0])


start_region, end_region = gcs.find_start_end_region(start_point, end_point, regions, vertices)
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

order = 5
continuity = 1
discrete_points_num = 10

# 调用贝塞尔曲线函数
Bezier_Curves = BezierCurves(best_path, regions, start_point, end_point, order, discrete_points_num, continuity)
control_points, all_curvatures = Bezier_Curves.solve()

# gcs.plot_2d()
# gcs.plot_relaxed_subgraph_2d()

# 创建第一个图形窗口
fig1, ax1 = plt.subplots(figsize=(10, 5))
Bezier_Curves.plot_bezier_curve_derivative(ax1, all_curvatures)

# 创建第二个图形窗口
fig2, ax2 = plt.subplots(figsize=(5, 5))
Bezier_Curves.plot_bezier_curve(ax2, control_points)

# 显示两个图形窗口
plt.show()