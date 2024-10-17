import cvxpy as cp
import numpy as np
import sys
import os
import math
import time

# 获取当前路径的父路径
parent_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(parent_path)
# 将父路径添加到 sys.path
sys.path.append(parent_path)

from graph_of_convex_sets import GraphOfConvexSets
from Calculate_Bezier import bezier_length, bezier_curve, bezier_kth_derivative
import matplotlib.pyplot as plt

def generate_constraints(qs, polygon):
    constraints = []
    n = len(polygon)
    for q in qs:
        for i in range(n):
            A = polygon[i]
            B = polygon[(i + 1) % n]
            constraint = (q[0] - A[0]) * (B[1] - A[1]) - (q[1] - A[1]) * (B[0] - A[0]) <= 0
            constraints.append(constraint)
    return constraints

def get_hexagon(outer_radius, inner_radius):
    """获取六边形的顶点"""
    def hexagon_vertices(center, radius):
        angles = np.linspace(0, 2 * np.pi, 7)
        return np.array([center + radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])

    # 大六边形的参数
    outer_radius = 2.5
    outer_center = np.array([0, 0])

    # 小六边形的参数
    inner_radius = 0.5
    inner_center = np.array([0, 0])

    # 计算大六边形和小六边形的顶点
    outer_hexagon = hexagon_vertices(outer_center, outer_radius)
    inner_hexagon = hexagon_vertices(inner_center, inner_radius)

    return outer_hexagon, inner_hexagon

gcs = GraphOfConvexSets()

order = 4
points_num = 100

outer_radius = 2.5
inner_radius = 0.5
outer_hexagon, inner_hexagon = get_hexagon(outer_radius, inner_radius)
# 定义梯形的四个顶点
A = []
B = []
C = []
D = []
for i in range(6):
    A.append(outer_hexagon[i])
    B.append(outer_hexagon[(i + 1) % 6])
    C.append(inner_hexagon[(i + 1) % 6])
    D.append(inner_hexagon[i])
polygons = [
    np.array([A[0], B[0], C[0], D[0]]),
    np.array([A[1], B[1], C[1], D[1]]),
    np.array([A[2], B[2], C[2], D[2]]),
    np.array([A[3], B[3], C[3], D[3]]),
    np.array([A[4], B[4], C[4], D[4]]),
    np.array([A[5], B[5], C[5], D[5]])
]

# print('polygons:', polygons)

# 创建顶点并添加约束
s_vertices = []
v1_vertices = []
v2_vertices = []
t_vertices = []
Qs = []
Qv1 = []
Qv2 = []
Qt = []
Xs = []
Xv1 = []
Xv2 = []
Xt = []

for k, polygon in enumerate(polygons):
    s = gcs.add_vertex(f"s_{k}")
    xs = [s.add_variable(2) for _ in range(order)]
    cs = np.array([1, 1])
    # if k == 2:
    #     s.add_constraint(xs[0] == [1, 1])
    #     for i in range(1, order):
    #         s.add_constraint(cp.abs(xs[i] - cs) <= [1, 1])
    # else:
    #     for i in range(order):
    #         s.add_constraint(cp.abs(xs[i] - cs) <= [1, 1])

    for i in range(order):
            s.add_constraint(cp.abs(xs[i] - cs) <= [1, 1])
    # 速度约束
    q0_s = 3 * (xs[1] - xs[0])
    q1_s = 3 * (xs[2] - xs[1])
    q2_s = 3 * (xs[3] - xs[2])
    qs = [q0_s, q1_s, q2_s]
    constraints_s = generate_constraints(qs, polygon)
    for constraint in constraints_s:
        s.add_constraint(constraint)
    s.add_cost(bezier_length(xs))

    v1 = gcs.add_vertex(f"v1_{k}")
    xv1 = [v1.add_variable(2) for _ in range(order)]
    cv1 = np.array([3, 0])
    for i in range(order):
        v1.add_constraint(cp.abs(xv1[i] - cv1) <= [1, 1])
    q0_v1 = 3 * (xv1[1] - xv1[0])
    q1_v1 = 3 * (xv1[2] - xv1[1])
    q2_v1 = 3 * (xv1[3] - xv1[2])
    qv1 = [q0_v1, q1_v1, q2_v1]
    constraints_v1 = generate_constraints(qv1, polygon)
    for constraint in constraints_v1:
        v1.add_constraint(constraint)
    v1.add_cost(bezier_length(xv1))

    v2 = gcs.add_vertex(f"v2_{k}")
    xv2 = [v2.add_variable(2) for _ in range(order)]
    cv2 = np.array([2, 3])
    for i in range(order):
        v2.add_constraint(cp.abs(xv2[i] - cv2) <= [1, 1])
    q0_v2 = 3 * (xv2[1] - xv2[0])
    q1_v2 = 3 * (xv2[2] - xv2[1])
    q2_v2 = 3 * (xv2[3] - xv2[2])
    qv2 = [q0_v2, q1_v2, q2_v2]
    constraints_v2 = generate_constraints(qv2, polygon)
    for constraint in constraints_v2:
        v2.add_constraint(constraint)
    v2.add_cost(bezier_length(xv2))

    t = gcs.add_vertex(f"t_{k}")
    xt = [t.add_variable(2) for _ in range(order)]
    ct = np.array([4, 2])
    # if k == 3:
    #     t.add_constraint(xt[-1] == [4, 1])
    #     for i in range(order - 1):
    #         t.add_constraint(cp.abs(xt[i] - ct) <= [1, 1])
    # else:
    #     for i in range(order):
    #         t.add_constraint(cp.abs(xt[i] - ct) <= [1, 1])

    for i in range(order):
            t.add_constraint(cp.abs(xt[i] - ct) <= [1, 1])
    q0_t = 3 * (xt[1] - xt[0])
    q1_t = 3 * (xt[2] - xt[1])
    q2_t = 3 * (xt[3] - xt[2])
    qt = [q0_t, q1_t, q2_t]
    constraints_t = generate_constraints(qt, polygon)
    for constraint in constraints_t:
        t.add_constraint(constraint)
    t.add_cost(bezier_length(xt))

    s_vertices.append(s)
    v1_vertices.append(v1)
    v2_vertices.append(v2)
    t_vertices.append(t)
    Qs.append(qs)
    Qv1.append(qv1)
    Qv2.append(qv2)
    Qt.append(qt)
    Xs.append(xs)
    Xv1.append(xv1)
    Xv2.append(xv2)
    Xt.append(xt)

# 遍历并打印每个顶点的名字
for vertex in gcs.vertices:
    print('Vertex name:', vertex.name)

# 添加二维空间内部边
# for i, s_1 in enumerate(s_vertices):
#     for j, s_2 in enumerate(s_vertices):
#         if abs(i - j) <=1 or (i == 0 and j == len(polygons) - 1) or (i == len(polygons) - 1 and j == 0): # 两个相邻的速度空间之间有边
#             if i != j:
#                 s_1_to_s_2 = gcs.add_edge(s_1, s_2)
#                 print(f"成功添加边{s_1.name} -> {s_2.name}")
#                 s_1_to_s_2.add_constraint(Xs[i][-1] == Xs[j][0]) # 二维约束
#                 s_1_to_s_2.add_constraint(Qs[i][-1] == Qs[j][0]) # 速度约束

# for i, v1_1 in enumerate(v1_vertices):
#     for j, v1_2 in enumerate(v1_vertices):
#         if abs(i - j) <=1 or (i == 0 and j == len(polygons) - 1) or (i == len(polygons) - 1 and j == 0):
#             if i != j:
#                 v1_1_to_v1_2 = gcs.add_edge(v1_1, v1_2)
#                 print(f"成功添加边{v1_1.name} -> {v1_2.name}")
#                 v1_1_to_v1_2.add_constraint(Xv1[i][-1] == Xv1[j][0])
#                 v1_1_to_v1_2.add_constraint(Qv1[i][-1] == Qv1[j][0])
    
# for i, v2_1 in enumerate(v2_vertices):
#     for j, v2_2 in enumerate(v2_vertices):
#         if abs(i - j) <=1 or (i == 0 and j == len(polygons) - 1) or (i == len(polygons) - 1 and j == 0):
#             if i != j:
#                 v2_1_to_v2_2 = gcs.add_edge(v2_1, v2_2)
#                 print(f"成功添加边{v2_1.name} -> {v2_2.name}")
#                 v2_1_to_v2_2.add_constraint(Xv2[i][-1] == Xv2[j][0])
#                 v2_1_to_v2_2.add_constraint(Qv2[i][-1] == Qv2[j][0])

# for i, t_1 in enumerate(t_vertices):
#     for j, t_2 in enumerate(t_vertices):
#         if abs(i - j) <=1 or (i == 0 and j == len(polygons) - 1) or (i == len(polygons) - 1 and j == 0):
#             if i != j:
#                 t_1_to_t_2 = gcs.add_edge(t_1, t_2)
#                 print(f"成功添加边{t_1.name} -> {t_2.name}")
#                 t_1_to_t_2.add_constraint(Xt[i][-1] == Xt[j][0])
#                 t_1_to_t_2.add_constraint(Qt[i][-1] == Qt[j][0])

# # 单向边
# for i, s_1 in enumerate(s_vertices):
#     for j, s_2 in enumerate(s_vertices):
#         # if j - i == 1 or (j == 0 and i == len(polygons) - 1):
#         if i - j == 1 or (i == 0 and j == len(polygons) - 1):
#             s_1_to_s_2 = gcs.add_edge(s_1, s_2)
#             print(f"成功添加边{s_1.name} -> {s_2.name}")
#             s_1_to_s_2.add_constraint(Xs[i][-1] == Xs[j][0]) # 二维约束
#             s_1_to_s_2.add_constraint(Qs[i][-1] == Qs[j][0]) # 速度约束

# for i, v1_1 in enumerate(v1_vertices):
#     for j, v1_2 in enumerate(v1_vertices):
#         # if j - i == 1 or (j == 0 and i == len(polygons) - 1):
#         if i - j == 1 or (i == 0 and j == len(polygons) - 1):
#             v1_1_to_v1_2 = gcs.add_edge(v1_1, v1_2)
#             print(f"成功添加边{v1_1.name} -> {v1_2.name}")
#             v1_1_to_v1_2.add_constraint(Xv1[i][-1] == Xv1[j][0])
#             v1_1_to_v1_2.add_constraint(Qv1[i][-1] == Qv1[j][0])

# for i, v2_1 in enumerate(v2_vertices):
#     for j, v2_2 in enumerate(v2_vertices):
#         # if j - i == 1 or (j == 0 and i == len(polygons) - 1):
#         if i - j == 1 or (i == 0 and j == len(polygons) - 1):
#             v2_1_to_v2_2 = gcs.add_edge(v2_1, v2_2)
#             print(f"成功添加边{v2_1.name} -> {v2_2.name}")
#             v2_1_to_v2_2.add_constraint(Xv2[i][-1] == Xv2[j][0])
#             v2_1_to_v2_2.add_constraint(Qv2[i][-1] == Qv2[j][0])

# for i, t_1 in enumerate(t_vertices):
#     for j, t_2 in enumerate(t_vertices):
#         # if j - i == 1 or (j == 0 and i == len(polygons) - 1):
#         if i - j == 1 or (i == 0 and j == len(polygons) - 1):
#             t_1_to_t_2 = gcs.add_edge(t_1, t_2)
#             print(f"成功添加边{t_1.name} -> {t_2.name}")
#             t_1_to_t_2.add_constraint(Xt[i][-1] == Xt[j][0])
#             t_1_to_t_2.add_constraint(Qt[i][-1] == Qt[j][0])


# 添加不同区域边
for i, s in enumerate(s_vertices):
    for j, v1 in enumerate(v1_vertices):
        # if abs(i - j) <=1 or (i == 0 and j == len(polygons) - 1) or (i == len(polygons) - 1 and j == 0):
        if i == j:
            sv1 = gcs.add_edge(s, v1)
            print(f"成功添加边{s.name} -> {v1.name}")
            sv1.add_constraint(Xs[i][-1] == Xv1[j][0])
            sv1.add_constraint(Qs[i][-1] == Qv1[j][0])
    for j, v2 in enumerate(v2_vertices):
        # if abs(i - j) <=1 or (i == 0 and j == len(polygons) - 1) or (i == len(polygons) - 1 and j == 0):
        if i == j:
            sv2 = gcs.add_edge(s, v2)
            print(f"成功添加边{s.name} -> {v2.name}")
            sv2.add_constraint(Xs[i][-1] == Xv2[j][0])
            sv2.add_constraint(Qs[i][-1] == Qv2[j][0])

for i, v1 in enumerate(v1_vertices):
    for j, t in enumerate(t_vertices):
        # if abs(i - j) <=1 or (i == 0 and j == len(polygons) - 1) or (i == len(polygons) - 1 and j == 0):
        if i == j:
            v1t = gcs.add_edge(v1, t)
            print(f"成功添加边{v1.name} -> {t.name}")
            v1t.add_constraint(Xv1[i][-1] == Xt[j][0])
            v1t.add_constraint(Qv1[i][-1] == Qt[j][0])

for i, v2 in enumerate(v2_vertices):
    for j, t in enumerate(t_vertices):
        # if abs(i - j) <=1 or (i == 0 and j == len(polygons) - 1) or (i == len(polygons) - 1 and j == 0):
        if i == j:
            v2t = gcs.add_edge(v2, t)
            print(f"成功添加边{v2.name} -> {t.name}")
            v2t.add_constraint(Xv2[i][-1] == Xt[j][0])
            v2t.add_constraint(Qv2[i][-1] == Qt[j][0])

gcs.graphviz()

# 添加虚拟起点重点区域
start_point = gcs.add_vertex("start_point")
x_start_point = start_point.add_variable(2)
start_point.add_constraint(x_start_point == [1,1])
for i, s in enumerate(s_vertices):
    start_point_to_s = gcs.add_edge(start_point, s)
    start_point_to_s.add_constraint(x_start_point == Xs[i][0])

end_point = gcs.add_vertex("end_point")
x_end_point = end_point.add_variable(2)
end_point.add_constraint(x_end_point == [4, 1.5])
for i, t in enumerate(t_vertices):
    t_to_end_point = gcs.add_edge(t, end_point)
    t_to_end_point.add_constraint(Xt[i][-1] == x_end_point)

start_time = time.time()
print('start_time:', start_time)
# prob = gcs.solve_shortest_path_relaxation(s_vertices[2], t_vertices[3])
prob = gcs.solve_shortest_path_relaxation(start_point, end_point)
end_time = time.time()
print('end_time:', end_time)
print('cost_time:', end_time - start_time)
print('Problem status:', prob.status)
print('Optimal value:', prob.value)

if prob.status == cp.OPTIMAL:
    P0 = [0] * 3
    P1 = [0] * 3
    P2 = [0] * 3
    P3 = [0] * 3
    Q0 = [0] * 3
    Q1 = [0] * 3
    Q2 = [0] * 3
    # 提取数值
    for i in range(len(s_vertices)):
        if Xs[i][0].value is not None:
            xs_values = np.array([var.value for var in Xs[i]])
            print('第 %d 个s:' % i, xs_values)
            P0[0] = xs_values[0]
            P1[0] = xs_values[1]
            P2[0] = xs_values[2]
            P3[0] = xs_values[3]
        if Xv1[i][0].value is not None:
            xv1_values = np.array([var.value for var in Xv1[i]])
            print('第 %d 个v1:' % i, xv1_values)
            P0[1] = xv1_values[0]
            P1[1] = xv1_values[1]
            P2[1] = xv1_values[2]
            P3[1] = xv1_values[3]
        if Xv2[i][0].value is not None:
            xv2_values = np.array([var.value for var in Xv2[i]])
            print('第 %d 个v2:' % i, xv2_values)
            P0[1] = xv2_values[0]
            P1[1] = xv2_values[1]
            P2[1] = xv2_values[2]
            P3[1] = xv2_values[3]
        if Xt[i][0].value is not None:
            xt_values = np.array([var.value for var in Xt[i]])
            print('第 %d 个t:' % i, xt_values)
            P0[2] = xt_values[0]
            P1[2] = xt_values[1]
            P2[2] = xt_values[2]
            P3[2] = xt_values[3]

    print('P0:', P0)
    print('P1:', P1)
    print('P2:', P2)
    print('P3:', P3)

    # 将列表转换为 NumPy 数组
    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)

    Q0 = 3 * (P1 - P0)
    Q1 = 3 * (P2 - P1)
    Q2 = 3 * (P3 - P2)

    print('Q0:', Q0)
    print('Q1:', Q1)
    print('Q2:', Q2)

    # 创建一个包含两个子图的窗口
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # 绘制第一个子图
    plt.sca(axs[0])
    gcs.plot_2d(is_plot_edges=False)
    gcs.plot_relaxed_subgraph_2d()
    # 绘制第二个子图
    plt.sca(axs[1])
    gcs.plot_velocity_2d(inner_radius = 0.5, outer_radius = 2.5)
    gcs.plot_subgraph_velocity_2d(Q0, Q1, Q2)

    plt.show()

else:
    print('No solution found')