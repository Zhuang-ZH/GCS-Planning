import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 定义控制点
control_points = np.array([
    [0, 0],
    [1, 2],
    [2, 3],
    [4, 0]
])

# 生成贝塞尔曲线
def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(num_points):
        for j in range(n + 1):
            curve[i] += control_points[j] * bernstein_poly(j, n, t[i])
    return curve

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


# 生成贝塞尔曲线
curve = bezier_curve(control_points)

# 测试 t=0 时的速度
t = 0
k = 1
derivative_at_t0 = bezier_kth_derivative(t, control_points, k)
print(f"Derivative at t={t}: {derivative_at_t0}")

# 测试 t=1 时的速度
t = 1
derivative_at_t1 = bezier_kth_derivative(t, control_points, k)
print(f"Derivative at t={t}: {derivative_at_t1}")

# 绘制贝塞尔曲线
plt.plot(curve[:, 0], curve[:, 1], label='Bezier Curve')
plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
plt.legend()
plt.show()