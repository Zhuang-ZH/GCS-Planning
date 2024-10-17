import cvxpy as cp
import numpy as np
import sys
import os

# 获取当前路径的父路径
parent_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(parent_path)
# 将父路径添加到 sys.path
sys.path.append(parent_path)
from graph_of_convex_sets import GraphOfConvexSets
import matplotlib.pyplot as plt

gcs = GraphOfConvexSets()

# L1 范数约束形成一个菱形（或正方形）。
# L2 范数约束形成一个圆（或球）。

s = gcs.add_vertex("s")
xs = s.add_variable(2)
cs = np.array([1, 0])
Ds = np.diag([2, 1])
# cp.norm(Ds @ (xs - cs), 2) <= 2 表示 xs 必须位于以 cs 为中心，半径为 2 的 L2 范数的椭圆内
s.add_constraint(cp.norm(Ds @ (xs - cs), 2) <= 2)

v1 = gcs.add_vertex("v1")
x1 = v1.add_variable(2)
c1 = np.array([4, 2])
# cp.norm(x1 - c1, np.inf) <= 1 表示 x1 必须位于以 c1 为中心，边长为 2 的轴对齐的正方形内
v1.add_constraint(cp.norm(x1 - c1, np.inf) <= 1)

v2 = gcs.add_vertex("v2")
x2 = v2.add_variable(2)
c2 = np.array([5.5, -2])
# cp.norm(x2 - c2, 1) <= 1.2 一个以 c2 为中心的菱形（或正方形，取决于维度），其边长为 2.4
v2.add_constraint(cp.norm(x2 - c2, 1) <= 1.2)
# cp.norm(x2 - c2, 2) <= 1 表示 x2 必须位于以 c2 为中心，半径为 1 的 L2 范数的球内
v2.add_constraint(cp.norm(x2 - c2, 2) <= 1)

v3 = gcs.add_vertex("v3")
x3 = v3.add_variable(2)
c3 = np.array([7, 2])
# cp.norm(x3 - c3, 2) <= 1 表示 x3 必须位于以 c3 为中心，半径为 1 的 L2 范数的球内
v3.add_constraint(cp.norm(x3 - c3, 2) <= 1)

t = gcs.add_vertex("t")
xt = t.add_variable(2)
ct = np.array([10, 0])
Dt = np.diag([1, 2])
t.add_constraint(cp.norm(Dt @ (xt - ct), 2) <= 2)
t.add_constraint(xt[0] <= ct[0])

sv1 = gcs.add_edge(s, v1)
sv1.add_cost(cp.norm(xs - x1, 2))
# 单调性约束
sv1.add_constraint(x1[1] >= xs[1])

sv2 = gcs.add_edge(s, v2)
sv2.add_cost(cp.norm(xs - x2, 2))
sv2.add_constraint(x2[1] >= xs[1])

v1v2 = gcs.add_edge(v1, v2)
v1v2.add_cost(cp.norm(x1 - x2, 2))
v1v2.add_constraint(x2[1] >= x1[1])

v1v3 = gcs.add_edge(v1, v3)
v1v3.add_cost(cp.norm(x1 - x3, 2))
v1v3.add_constraint(x3[1] >= x1[1])

v2t = gcs.add_edge(v2, t)
v2t.add_cost(cp.norm(x2 - xt, 2))
v2t.add_constraint(xt[1] >= x2[1])

v2v3 = gcs.add_edge(v2, v3)
v2v3.add_cost(cp.norm(x2 - x3, 2))
v2v3.add_constraint(x3[1] >= x2[1])

v3t = gcs.add_edge(v3, t)
v3t.add_cost(cp.norm(x3 - xt, 2))
v3t.add_constraint(xt[1] >= x3[1])

gcs.graphviz()

# prob = gcs.solve_shortest_path(s, t)
prob = gcs.solve_shortest_path_relaxation(s, t)
print('Problem status:', prob.status)
print('Optimal value:', prob.value)

plt.figure()
plt.gca().set_aspect('equal')
plt.axis('on')

gcs.plot_2d()
# gcs.plot_subgraph_2d()
gcs.plot_relaxed_subgraph_2d()
plt.show()
# plt.savefig('shortest_path.pdf')