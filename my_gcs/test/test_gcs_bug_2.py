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

s = gcs.add_vertex("s")
xs = s.add_variable(2)
cs = np.array([0, 1]) # center
size_s = np.array([1.5, 1]) / 2 # size
s.add_constraint(cp.abs(xs - cs) <= size_s)

v1 = gcs.add_vertex("v1")
xv1 = v1.add_variable(2)
cv1 = np.array([0, -0.75])
size_v1 = np.array([1, 1]) / 2
v1.add_constraint(cp.abs(xv1 - cv1) <= size_v1)

t = gcs.add_vertex("t")
xt = t.add_variable(2)
ct = np.array([0, -2.5])
size_t = np.array([1.5, 1.0]) / 2
t.add_constraint(cp.abs(xt - ct) <= size_t)

sv1 = gcs.add_edge(s, v1)
sv1.add_cost(cp.norm(xs - xv1, 2))
sv1.add_constraint(xv1[1] >= xs[1])

v1t = gcs.add_edge(v1, t)
v1t.add_cost(cp.norm(xv1 - xt, 2))
v1t.add_constraint(xt[1] >= xv1[1])

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
# gcs.plot_relaxed_subgraph_2d()
plt.show()
# plt.savefig('shortest_path.pdf')