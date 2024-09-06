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
from segment_image import generate_regions, generate_edges

# 定义地图范围
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# 定义起点和终点
start_point = np.array([-1.9, 0])
end_point = np.array([1.9, 0])

order = 3
continuity = 1
discrete_points_num = 10


gcs = GraphOfConvexSets()
expension_gcs = GraphOfConvexSets()

obstacles = [
    {"name": "ob1", "center": [-1.25, -0.5], "size": [0.5, 2.0]},
    {"name": "ob2", "center": [-0.75, 1.25], "size": [0.5, 0.5]},
    {"name": "ob3", "center": [0.75, 0.5], "size": [1.5, 1.0]},
    {"name": "ob4", "center": [1.25, -0.75], "size": [0.5, 0.5]},
]

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
regions = generate_regions(x_min, x_max, y_min, y_max, obstacles)
expension_regions = generate_regions(x_min, x_max, y_min, y_max, expansion_obstacles)

vertices = gcs.add_region_to_GCS(regions)
expension_vertices = expension_gcs.add_region_to_GCS(expension_regions)

edges = generate_edges(regions)
expension_edges = generate_edges(expension_regions)

gcs.add_edge_to_GCS(edges, vertices)
expension_gcs.add_edge_to_GCS(expension_edges, expension_vertices)

gcs.graphviz()
expension_gcs.graphviz()


start_region, end_region = gcs.find_start_end_region(start_point, end_point, regions)
expension_start_region, expension_end_region = expension_gcs.find_start_end_region(start_point, end_point, expension_regions)

s = vertices[start_region][0]
t = vertices[end_region][0]
expension_s = expension_vertices[expension_start_region][0]
expension_t = expension_vertices[expension_end_region][0]

prob = gcs.solve_shortest_path_relaxation(s, t)
expension_prob = expension_gcs.solve_shortest_path_relaxation(expension_s, expension_t)

print('Problem status:', prob.status)
print('Optimal value:', prob.value)

# 获取边权重
edge_weights = gcs.get_edge_weights(relaxed=True)
expension_edge_weights = expension_gcs.get_edge_weights(relaxed=True)

paths = randomForwardPathSearch(regions, edges, edge_weights, start_region, end_region, max_paths=10, max_trials=100, seed=5)
expension_paths = randomForwardPathSearch(expension_regions, expension_edges, expension_edge_weights, expension_start_region, expension_end_region, max_paths=10, max_trials=100, seed=5)

best_path = findMaxWeightPath(paths)
expension_best_path = findMaxWeightPath(expension_paths)

print(best_path)

# 调用线性规划函数
# points = linear(best_path, regions, start_point, end_point)

# 调用贝塞尔曲线函数
Bezier_Curves = BezierCurves(best_path, regions, start_point, end_point, order, discrete_points_num, continuity)
expension_Bezier_Curves = BezierCurves(expension_best_path, expension_regions, start_point, end_point, order, discrete_points_num, continuity)

control_points, all_curvatures, curve_points_all= Bezier_Curves.solve()
expension_control_points, expension_all_curvatures, expension_curve_points_all = expension_Bezier_Curves.solve()


# gcs.plot_2d()
# gcs.plot_relaxed_subgraph_2d()

# 创建第一个图形窗口
fig1, ax1 = plt.subplots(figsize=(10, 5))
Bezier_Curves.plot_bezier_curve_derivative(ax1, all_curvatures)

# 创建第二个图形窗口
fig2, ax2 = plt.subplots(figsize=(5, 5))
Bezier_Curves.plot_bezier_curve(ax2, control_points)
expension_Bezier_Curves.plot_bezier_curve(ax2, expension_control_points)

# 显示两个图形窗口
plt.show()