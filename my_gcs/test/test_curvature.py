import cvxpy as cp
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def bezier_curve(t, control_points):
    control_points = np.array(control_points)
    n = len(control_points) - 1
    curve_point = np.zeros(2)
    for i in range(n + 1):
        binomial_coeff = np.math.comb(n, i)
        curve_point = curve_point + binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * control_points[i]
    return curve_point

def bezier_curve_kth_derivative(t, control_points, k):
    n = len(control_points) - 1
    if k > n:
        print("k is greater than the number of control points")
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

def bezier_curve_curvature(t, control_points):
    
    first_derivative = bezier_curve_kth_derivative(t, control_points, 1)
    second_derivative = bezier_curve_kth_derivative(t, control_points, 2)
    
    # 正数为向上弯曲，负数为向下弯曲
    numerator = first_derivative[0] * second_derivative[1] - first_derivative[1] * second_derivative[0]
    denominator = cp.norm(first_derivative) ** 3
    
    epsilon = 1e-6
    curvature = numerator / (denominator + epsilon)

    return curvature

def calculate_curvature(points):
    # 计算每个点的导数
    dx_dt = np.gradient(points[:, 0])
    dy_dt = np.gradient(points[:, 1])

    print(f"dx_dt[-1]: {dx_dt[-1]}")
    print(f"dy_dt[-1]: {dy_dt[-1]}")
    
    # 计算每个点的二阶导数
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    print(f"d2x_dt2[-1]: {d2x_dt2[-1]}")
    print(f"d2y_dt2[-1]: {d2y_dt2[-1]}")
    
    # 计算曲率
    curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt**2 + dy_dt**2)**1.5
    
    return curvature
    
def plot_bezier_curve(ax, uniform_points, control_points):
    for name, points in uniform_points.items():
        ax.plot(points[:, 0], points[:, 1], 'o', markersize=1)
    for name, points in control_points.items():
        for point in points:
            if point is not None:
                ax.plot(point[0], point[1], 'bo')
    ax.grid(True)

    # ax.text(control_points["curve1"][1][0], control_points["curve1"][1][1], "curve1", fontsize=12)
    # ax.text(control_points["curve2"][1][0], control_points["curve2"][1][1], "curve2", fontsize=12)
    # ax.text(control_points["curve3"][1][0], control_points["curve3"][1][1], "curve3", fontsize=12)
    # ax.text(control_points["curve4"][1][0], control_points["curve4"][1][1], "curve4", fontsize=12)

def plot_bezier_curve_curvature(ax, all_curvatures):
    # 绘制曲率值折线图
    ax.plot(all_curvatures, marker='o')
    ax.set_xlabel('points')
    ax.set_ylabel('curvatures')
    ax.set_title('curvatures-points')

    # 根据数据范围调整横坐标和纵坐标的最小单位
    ax.xaxis.set_major_locator(MultipleLocator(max(1, len(all_curvatures) // 10)))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.grid(True)


control_points = {"curve1": np.array([[0,1], [0.5523,1], [1, 0.5523], [1,0]]),
                  "curve2": np.array([[1,0], [1, -1.1046], [1.8954, -2], [3,-2]]),
                  "curve3": np.array([[3,-2], [3.5523, -2], [4, -1.5523], [4,-1]]),
                  "curve4": np.array([[4,-1], [4, 0.1046], [4.8954, 1], [6,1]]),}

num_points = 100
t_values = np.linspace(0, 1, 100)

uniform_points = {}
uniform_points["curve1"] = np.array([bezier_curve(t, control_points["curve1"]) for t in t_values])
uniform_points["curve2"] = np.array([bezier_curve(t, control_points["curve2"]) for t in t_values])
uniform_points["curve3"] = np.array([bezier_curve(t, control_points["curve3"]) for t in t_values])
uniform_points["curve4"] = np.array([bezier_curve(t, control_points["curve4"]) for t in t_values])
uniform_points_all = np.concatenate(list(uniform_points.values()))

curvatures = []
curvatures.extend(bezier_curve_curvature(t, control_points["curve1"]).value for t in t_values)
curvatures.extend(bezier_curve_curvature(t, control_points["curve2"]).value for t in t_values)
curvatures.extend(bezier_curve_curvature(t, control_points["curve3"]).value for t in t_values)
curvatures.extend(bezier_curve_curvature(t, control_points["curve4"]).value for t in t_values)
print(f"curvatures: {curvatures}")

# points_all = np.concatenate(list(uniform_points.values()))
# curvatures = calculate_curvature(points_all)

fig1, ax1 = plt.subplots(figsize=(10, 5))
plot_bezier_curve(ax1, uniform_points, control_points)

fig2, ax2 = plt.subplots(figsize=(10, 5))
plot_bezier_curve_curvature(ax2, curvatures)

plt.show()