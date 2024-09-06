import time
import numpy as np
import matplotlib.pyplot as plt

from pydrake.geometry.optimization import HPolyhedron
from pydrake.solvers import MosekSolver
from pydrake.trajectories import PiecewisePolynomial

from gcs.bezier import BezierGCS
from gcs.rounding import *

from plot_obstacle import Plotting
import env
from datetime import datetime

x_start = np.array([-1.9, 0])
x_goal = np.array([1.9, 0])

regions = [
    HPolyhedron.MakeBox([-2.0, -2.0],[-1.5, 2.0]), 
    HPolyhedron.MakeBox([-1.5, 0.5],[-1.0, 2.0]), 
    HPolyhedron.MakeBox([-1.5, -2.0],[-1.0, -1.5]),   
    HPolyhedron.MakeBox([-1.0, 1.5],[-0.5, 2.0]), 
    HPolyhedron.MakeBox([-1.0, -2.0],[-0.5, 1.0]), 
    HPolyhedron.MakeBox([-0.5, -2.0],[0.0, 2.0]), 
    HPolyhedron.MakeBox([0.0, 1.0],[1.5, 2.0]),   
    HPolyhedron.MakeBox([0.0, -2.0],[1.0, 0.0]), 
    HPolyhedron.MakeBox([1.0, -0.5],[1.5, 0.0]), 
    HPolyhedron.MakeBox([1.0, -2.0],[1.5, -1.0]), 
    HPolyhedron.MakeBox([1.5, -2.0],[2.0, 2.0]),   
]

# 创建一个figure和axes对象
fig, ax = plt.subplots()

obstacle = Plotting(x_start, x_goal,ax)
obstacle.plot_grid("gcs")

# 设置了图形的坐标轴比例，使得单位长度在 x 轴和 y 轴上的显示比例相同
ax.set_aspect('equal')
# 分别设置了当前坐标轴的 x 轴和 y 轴的显示范围
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-2.1, 2.1)

# 记录开始时间
start_time = time.time()

# # 贝塞尔轨迹
# # regions定义了贝塞尔曲线可以存在的空间
# # order是贝塞尔曲线的阶数,阶数为 5 意味着贝塞尔曲线将有 6 个控制点
# # continuity是贝塞尔曲线的连续性，连续性为 2 意味着贝塞尔曲线的一阶导数是连续的
# b_gcs = BezierGCS(regions, order=6, continuity=2)
# # 添加时间成本
# b_gcs.addTimeCost(1)
# # 添加路径长度成本
# b_gcs.addPathLengthCost(1)
# # 添加起始点和目标点
# b_gcs.addSourceTarget(x_start, x_goal)
# # 设置求解器
# b_gcs.setSolver(MosekSolver())
# # 求解路径
# bezier_traj = b_gcs.SolvePath(rounding=True, verbose=True)[0]

qdot_min = -1
qdot_max = 1
velocity = np.zeros((2, 2))
order = 6
continuity = 2
hdot_min = 1e-1
regularizer = [1e-1,1e-1]
relaxation = True
gcs = BezierGCS(regions, order, continuity, hdot_min=hdot_min)
gcs.setSolver(MosekSolver())
gcs.setPaperSolverOptions()
gcs.addTimeCost(1)
gcs.addDerivativeRegularization(*regularizer, 2)
gcs.addVelocityLimits([qdot_min] * 2, [qdot_max] * 2)
gcs.addSourceTarget(x_start, x_goal, velocity=velocity)
bezier_traj = gcs.SolvePath(rounding=relaxation, verbose=True)[0]
# # 两段代码的主要区别在于贝塞尔曲线的阶数、连续性、速度限制和正则化参数的设置，这些因素会影响生成路径的形状。

# 记录结束时间
end_time = time.time()
# 计算总的计算时间
total_time = end_time - start_time
print(f"凸优化总计算时间: {total_time:.2f} 秒")

# 生成一个等间隔的数值数组，用于绘制贝塞尔轨迹
samples = np.linspace(bezier_traj.start_time(), bezier_traj.end_time(),500)
# 从贝塞尔轨迹中获取样本点的值
values = np.squeeze([bezier_traj.value(s) for s in samples])

# 绘制贝塞尔轨迹
ax.plot(values[:, 0], values[:, 1], "c-")
plt.show()

