import numpy as np
import cvxpy as cp
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

def hexagon_vertices(center, radius):
    """计算六边形的顶点"""
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

# 绘制大六边形和小六边形
plt.figure()
plt.plot(outer_hexagon[:, 0], outer_hexagon[:, 1], 'b-', label='Outer Hexagon')
plt.plot(inner_hexagon[:, 0], inner_hexagon[:, 1], 'r-', label='Inner Hexagon')

# 分割剩余区域并绘制梯形
for i in range(6):
    trapezoid = np.array([outer_hexagon[i], outer_hexagon[i+1], inner_hexagon[i+1], inner_hexagon[i]])
    plt.fill(trapezoid[:, 0], trapezoid[:, 1], edgecolor='black', fill=False)
    for vertex in trapezoid:
        plt.text(vertex[0], vertex[1], f'({vertex[0]:.2f}, {vertex[1]:.2f})', fontsize=8, ha='right')

# 设置图形属性
plt.gca().set_aspect('equal')
plt.legend()
plt.grid(True)
plt.title('Hexagon with Inner Hexagon and Trapezoids')
# plt.show()

print('Outer Hexagon:', outer_hexagon)
print('Inner Hexagon:', inner_hexagon)

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

qs = np.array([
    [0.5, 0.5],
    [0.75, 0.75],
    [0.25, 0.25]
])
polygon = polygons[0]
# 生成约束
constraints = generate_constraints(qs, polygon)

# 绘制多边形
# plt.figure()
plt.plot(*polygon.T, 'b-', label='Polygon')
plt.fill(*polygon.T, 'b', alpha=0.1)

# 绘制速度约束点
for q in qs:
    plt.plot(q[0], q[1], 'ro')

# 绘制约束的凸区域
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Convex Region Defined by Constraints')
plt.legend()
plt.grid(True)
plt.show()