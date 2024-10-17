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


# 计算贝塞尔曲线的长度
def bezier_length(control_points, points_num=100):
    t = np.linspace(0, 1, points_num)
    curve = [sum(control_points[j] * bernstein_poly(j, len(control_points) - 1, t[i]) for j in range(len(control_points))) for i in range(points_num)]
    length = sum(cp.norm(curve[i+1] - curve[i], 2) for i in range(points_num - 1))
    return length

# 伯恩斯坦多项式
def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

# 组合数
def comb(n, k):
    return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))

def bezier_kth_derivative(t, control_points, k):
    n = len(control_points) - 1
    if k > n:
        return np.zeros(2)  # 如果 k 大于控制点的数量，返回零向量
    
    derivative = np.zeros(2)
    
    for i in range(n - k + 1):
        binomial_coeff = np.math.comb(n - k, i)
        term = binomial_coeff * ((1 - t) ** (n - k - i)) * (t ** i)
        
        # 计算k阶导数的系数 (n-k)(n-k-1)...(n-k-(k-1))
        for j in range(k):
            term = term * (n - j)
        
        # 计算贝塞尔曲线的k阶导数，应用差分形式
        point_diff = np.zeros(2)
        for l in range(k + 1):
            sign = (-1) ** l
            binom = np.math.comb(k, l)
            point_diff = point_diff + sign * binom * control_points[i + k - l]
        
        derivative = derivative + term * point_diff
    
    return derivative

gcs = GraphOfConvexSets()

order = 4
points_num = 100

s = gcs.add_vertex("s")
xs = [s.add_variable(2) for _ in range(order)]
cs = np.array([1, 1])
s.add_constraint(xs[0] == [0, 0])
for i in range(1, order):
    s.add_constraint(cp.abs(xs[i] - cs) <= [1, 1])
s.add_cost(bezier_length(xs))

v1 = gcs.add_vertex("v1")
xv1 = [v1.add_variable(2) for _ in range(order)]
cv1 = np.array([3, 0])
for i in range(order):
    v1.add_constraint(cp.abs(xv1[i] - cv1) <= [1, 1])
v1.add_cost(bezier_length(xv1))

v2 = gcs.add_vertex("v2")
xv2 = [v2.add_variable(2) for _ in range(order)]
cv2 = np.array([2, 3])
for i in range(order):
    v2.add_constraint(cp.abs(xv2[i] - cv2) <= [1, 1])
v2.add_cost(bezier_length(xv2))

t = gcs.add_vertex("t")
xt = [t.add_variable(2) for _ in range(order)]
ct = np.array([4, 2])
t.add_constraint(xt[-1] == [4, 3])
for i in range(order - 1):
    t.add_constraint(cp.abs(xt[i] - ct) <= [1, 1])
t.add_cost(bezier_length(xt))

sv1 = gcs.add_edge(s, v1)
v1t = gcs.add_edge(v1, t)
sv2 = gcs.add_edge(s, v2)
v2t = gcs.add_edge(v2, t)


# 二维坐标约束
sv1.add_constraint(xs[-1] == xv1[0])
v1t.add_constraint(xv1[-1] == xt[0])
sv2.add_constraint(xs[-1] == xv2[0])
v2t.add_constraint(xv2[-1] == xt[0])

# 速度约束
sv1.add_constraint(bezier_kth_derivative(1, xs, 1) == bezier_kth_derivative(0, xv1, 1))
v1t.add_constraint(bezier_kth_derivative(1, xv1, 1) == bezier_kth_derivative(0, xt, 1))
sv2.add_constraint(bezier_kth_derivative(1, xs, 1) == bezier_kth_derivative(0, xv2, 1))
v2t.add_constraint(bezier_kth_derivative(1, xv2, 1) == bezier_kth_derivative(0, xt, 1))

v = 1.0  
v_max = 2.5 # 最大速度值
v_min = 0.5 # 最小速度值
for i in range(points_num):
    t_bezier = i / (points_num - 1)
    sv1.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xs, 1) * v, 2) <= v_max)
    sv1.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xv1, 1) * v, 2) <= v_max)
    v1t.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xv1, 1) * v, 2) <= v_max)
    v1t.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xt, 1) * v, 2) <= v_max)
    sv2.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xs, 1) * v, 2) <= v_max)
    sv2.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xv2, 1) * v, 2) <= v_max)
    v2t.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xv2, 1) * v, 2) <= v_max)
    v2t.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xt, 1) * v, 2) <= v_max)


# # 加速度约束
# sv1.add_constraint(bezier_kth_derivative(1, xs, 2) == bezier_kth_derivative(0, xv1, 2))
# v1t.add_constraint(bezier_kth_derivative(1, xv1, 2) == bezier_kth_derivative(0, xt, 2))
# sv2.add_constraint(bezier_kth_derivative(1, xs, 2) == bezier_kth_derivative(0, xv2, 2))
# v2t.add_constraint(bezier_kth_derivative(1, xv2, 2) == bezier_kth_derivative(0, xt, 2))

# a_max = 2.0 # 最大速度值
# for i in range(points_num):
#     t_bezier = i / (points_num - 1)
#     sv1.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xs, 2), 2) <= a_max)
#     sv1.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xv1, 2), 2) <= a_max)
#     v1t.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xv1, 2), 2) <= a_max)
#     v1t.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xt, 2), 2) <= a_max)
#     sv2.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xs, 2), 2) <= a_max)
#     sv2.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xv2, 2), 2) <= a_max)
#     v2t.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xv2, 2), 2) <= a_max)
#     v2t.add_constraint(cp.norm(bezier_kth_derivative(t_bezier, xt, 2), 2) <= a_max)



gcs.graphviz()

prob = gcs.solve_shortest_path_relaxation(s, t)
print('Problem status:', prob.status)
print('Optimal value:', prob.value)


# 提取数值
xs_values = np.array([var.value for var in xs])
if xv1[0].value is not None:
    xv1_values = np.array([var.value for var in xv1])
if xv2[0].value is not None:
    xv2_values = np.array([var.value for var in xv2])
xt_values = np.array([var.value for var in xt])
# 初始化列表以存储范数
norms_xs = []
norms_xv1 = []
norms_xv2 = []
norms_xt = []

for t in np.linspace(0, 1, 50):
    # 计算导数
    derivative_xs = bezier_kth_derivative(t, xs_values, 1)
    norms_xs.append(np.linalg.norm(derivative_xs))
    if xv1[0].value is not None:
        derivative_xv1 = bezier_kth_derivative(t, xv1_values, 1)
        norms_xv1.append(np.linalg.norm(derivative_xv1))
    if xv2[0].value is not None:
        derivative_xv2 = bezier_kth_derivative(t, xv2_values, 1)
        norms_xv2.append(np.linalg.norm(derivative_xv2))
    derivative_xt = bezier_kth_derivative(t, xt_values, 1)
    norms_xt.append(np.linalg.norm(derivative_xt))

# 打印结果
# print("Norms for xs_values:", norms_xs)
# print("Norms for xv1_values:", norms_xv1)
# print("Norms for xt_values:", norms_xt)

print("Max Norms for xs_values:", max(norms_xs))
print("Minimum Norms for xs_values:", min(norms_xs))
if xv1[0].value is not None:
    print("Max Norms for xv1_values:", max(norms_xv1))
    print("Minimum Norms for xv1_values:", min(norms_xv1))
if xv2[0].value is not None:
    print("Max Norms for xv2_values:", max(norms_xv2))
    print("Minimum Norms for xv2_values:", min(norms_xv2))
print("Max Norms for xt_values:", max(norms_xt))
print("Minimum Norms for xt_values:", min(norms_xt))




plt.figure()
plt.gca().set_aspect('equal')
plt.axis('on')

gcs.plot_2d(is_plot_edges=False)
# gcs.plot_subgraph_2d()
gcs.plot_relaxed_subgraph_2d()
plt.show()
# plt.savefig('shortest_path.pdf')