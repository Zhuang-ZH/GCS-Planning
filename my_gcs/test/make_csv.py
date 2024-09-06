import cvxpy as cp
import numpy as np
import sys
import pandas as pd


sys.path.append('../PathPlanning/GCS_Planning/my_gcs')
from graph_of_convex_sets import GraphOfConvexSets, Vertex
import matplotlib.pyplot as plt
from RandomDFS_relaxed import randomForwardPathSearch, findMaxWeightPath
from find_points import linear
from BezierCurves import BezierCurves
from matplotlib.ticker import MultipleLocator
from segment_image import generate_regions, generate_edges
from Generate_Track import generate_track_control_points

# 定义地图范围
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# 定义起点和终点
start_point = np.array([-1.8, 0])
end_point = np.array([0.7, 1.2])

order = 3
continuity = 1
discrete_points_num = 100

gcs = GraphOfConvexSets()

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

edges = generate_edges(regions)

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

paths = randomForwardPathSearch(regions, edges, edge_weights, start_region, end_region, max_paths=10, max_trials=100, seed=5)

best_path = findMaxWeightPath(paths)

print(best_path)

# 调用线性规划函数
# points = linear(best_path, regions, start_point, end_point)

# 调用贝塞尔曲线函数
Bezier_Curves = BezierCurves(best_path, regions, start_point, end_point, order, discrete_points_num, continuity)

control_points, all_curvatures, curve_points, all_speed, uniform_points, accerlation, headings= Bezier_Curves.solve()
print(f"control_points: {control_points}")

# 按时间离散
all_points = np.concatenate(list(curve_points.values()))
# print(f"all_points: {all_points}")
# 按空间离散
# all_points = np.concatenate(list(uniform_points.values()))
# print(f"all_points: {all_points}")

# all_curvatures = BezierCurves.calculate_curvature(all_points)
# all_speed = np.full(len(all_points), 0.5)

# print(f"all_points: {all_points[0:20]}")
# print(f"all_curvatures: {all_curvatures}")

# 创建 DataFrame
data = {
    'x': all_points[:, 0],
    'y': all_points[:, 1],
    'ψ': headings,
    'v': all_speed ,  # v 设置为 0.5
    'κ': all_curvatures,
    'δ': np.full(len(all_points), 0.0)      # δ 设置为 0
}

df = pd.DataFrame(data)

# 输出到 CSV 文件
df.to_csv('GCS_Planning/my_gcs/test/output.csv', index=False)

# # 曲率折线图
# fig1, ax1 = plt.subplots(figsize=(10, 5))
# Bezier_Curves.plot_bezier_curve_curvature(ax1, all_curvatures)

# # 路径 & map
# fig2, ax2 = plt.subplots(figsize=(5, 5))
# Bezier_Curves.plot_regions(ax2)
# # 按时间离散
# Bezier_Curves.plot_bezier_curve(ax2, curve_points, control_points)
# # 按空间离散
# # Bezier_Curves.plot_bezier_curve(ax2, uniform_points, control_points)

# # 速度折线图
# fig3, ax3 = plt.subplots(figsize=(10, 5))
# Bezier_Curves.plot_bezier_curve_speed(ax3, all_speed)

# # 加速度折线图
# fig4, ax4 = plt.subplots(figsize=(10, 5))
# Bezier_Curves.plot_bezier_curve_acceleration(ax4, accerlation)

# # 航向角折线图
# fig5, ax5 = plt.subplots(figsize=(10, 5))
# Bezier_Curves.plot_bezier_curve_headings(ax5, headings)

# # 显示两个图形窗口
# plt.show()

# 路径 & map
fig1, ax1 = plt.subplots(figsize=(5, 5))
Bezier_Curves.plot_regions(ax1)
# 按时间离散
Bezier_Curves.plot_bezier_curve(ax1, curve_points, control_points)
# 按空间离散
# Bezier_Curves.plot_bezier_curve(ax2, uniform_points, control_points)

# 创建一个 2x2 的子图布局
fig, axs = plt.subplots(4, 1, figsize=(8, 12))

# 绘制曲率折线图
Bezier_Curves.plot_bezier_curve_curvature(axs[0], all_curvatures)
axs[0].set_title("curvature")

# 绘制速度折线图
Bezier_Curves.plot_bezier_curve_speed(axs[1], all_speed)
axs[1].set_title("speed")

# 绘制加速度折线图
Bezier_Curves.plot_bezier_curve_acceleration(axs[2], accerlation)
axs[2].set_title("acceleration")

# 绘制航向角折线图
Bezier_Curves.plot_bezier_curve_headings(axs[3], headings)
axs[3].set_title("headings")

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()