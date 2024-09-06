import cvxpy as cp
import numpy as np
import sys

sys.path.append('../PathPlanning/GCS_Planning/my_gcs')
from graph_of_convex_sets import GraphOfConvexSets
import matplotlib.pyplot as plt

gcs = GraphOfConvexSets()

s = gcs.add_vertex("s")
xs = s.add_variable(2)
cs = np.array([-1.25, 1.25])
size_s = np.array([0.5, 1.5]) / 2
s.add_constraint(cp.abs(xs - cs) <= size_s)

t = gcs.add_vertex("t")
xt = t.add_variable(2)
ct = np.array([0.5, -1.0])
size_t = np.array([1.0, 1.0]) / 2
t.add_constraint(cp.abs(xt - ct) <= size_t)

st = gcs.add_edge(s, t)
st.add_cost(cp.norm(xs - xt, 2))
st.add_constraint(xt[1] >= xs[1])

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