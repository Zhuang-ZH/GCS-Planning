import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class BezierCurves:
    def __init__(self, best_path, regions, start_point, end_point, order, discrete_points_num, continuity=1):
        self.best_path = best_path
        self.regions = regions
        self.start_point = start_point
        self.end_point = end_point
        self.order = order
        self.continuity = continuity
        self.discrete_points_num = discrete_points_num

    @staticmethod
    def bezier_curve(t, control_points):
        control_points = np.array(control_points)
        n = len(control_points) - 1
        curve_point = np.zeros(2)
        for i in range(n + 1):
            binomial_coeff = np.math.comb(n, i)
            curve_point += binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * control_points[i]
        return curve_point

    @staticmethod
    def bezier_curve_derivative(t, control_points):
        control_points = np.array(control_points)  # 转换为 NumPy 数组
        n = len(control_points) - 1
        derivative = np.zeros(2)
        for i in range(n):
            binomial_coeff = np.math.comb(n - 1, i)
            derivative += n * binomial_coeff * (t ** i) * ((1 - t) ** (n - 1 - i)) * (control_points[i + 1] - control_points[i])
        return derivative

    def arc_length(self, control_points):
        def integrand(t):
            derivative = self.bezier_curve_derivative(t, control_points)
            return np.linalg.norm(derivative)
        
        length, _ = quad(integrand, 0, 1)
        return length

    def find_t_for_arc_length(self, control_points, target_length):
        total_length = self.arc_length(control_points)
        
        def equation(t):
            length, _ = quad(lambda x: np.linalg.norm(self.bezier_curve_derivative(x, control_points)), 0, t)
            return length - target_length
        
        t_value, = fsolve(equation, 0.5)
        return t_value

    def calculate_uniform_bezier_points(self, control_points, discrete_points_num=100):
        total_length = self.arc_length(control_points)
        uniform_arc_lengths = np.linspace(0, total_length, discrete_points_num)
        uniform_points = [self.bezier_curve(self.find_t_for_arc_length(control_points, s), control_points) for s in uniform_arc_lengths]
        uniform_points[0] = control_points[0]  # 确保起点等于第一个控制点
        uniform_points[-1] = control_points[-1]  # 确保终点等于最后一个控制点
        return np.array(uniform_points)

# 示例使用
control_points = np.array([[-1.5,  0.5],[-1.36572986,  0.6125], [-1.053073,  0.6125],[-1.45343033,  0.6125],[-1.2240594,  0.6125],[-1. , 0.5]])
# control_points = np.array([[-1.5,  0.5],[-1.45343033,  0.6125],[-1.36572986,  0.6125],[-1.2240594,  0.6125], [-1.053073,  0.6125],[-1. , 0.5]])
bezier = BezierCurves(None, None, None, None, None, None)
uniform_points = bezier.calculate_uniform_bezier_points(control_points, discrete_points_num=100)

print(uniform_points)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(control_points[:, 0], control_points[:, 1], 'bo', markersize=1)
ax.plot(uniform_points[:, 0], uniform_points[:, 1], 'ro', markersize=1)
plt.show()