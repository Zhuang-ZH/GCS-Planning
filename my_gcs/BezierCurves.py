import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.integrate import quad
from scipy.optimize import fsolve
from matplotlib.ticker import MultipleLocator

class BezierCurves:
    def __init__(self, best_path, regions, start_point, end_point, order, discrete_points_num, continuity=1):
        self.best_path = best_path
        self.regions = regions
        self.start_point = start_point
        self.end_point = end_point
        self.order = order
        self.continuity = continuity
        self.discrete_points_num = discrete_points_num

    @ staticmethod
    def bezier_curve(t, control_points):
        control_points = np.array(control_points)
        n = len(control_points) - 1
        curve_point = np.zeros(2)
        for i in range(n + 1):
            binomial_coeff = np.math.comb(n, i)
            curve_point = curve_point + binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * control_points[i]
        return curve_point
    
    @ staticmethod
    def calculate_bezier_points(control_points_dict, discrete_points_num=100):
        all_points = []
        for key, control_points in control_points_dict.items():
            for t in np.linspace(0, 1, discrete_points_num):
                point = BezierCurves.bezier_curve(t, control_points)
                all_points.append(point)
        return np.vstack(all_points)

    @ staticmethod
    def calculate_curvature(points):
        # 计算每个点的导数
        dx_dt = np.gradient(points[:, 0])
        dy_dt = np.gradient(points[:, 1])
        
        # 计算每个点的二阶导数
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        
        # 计算曲率
        curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt**2 + dy_dt**2)**1.5
        
        return curvature
    
    @ staticmethod
    def calculate_speed(points):
        # 计算每个点与下一个点之间的差值
        deltas = np.diff(points, axis=0)
        # 计算每个点的速度
        speed = np.linalg.norm(deltas, axis=1)
        # 添加最后一个点的速度，假设与倒数第二个点相同
        speed = np.append(speed, speed[-1])
        return speed
    
    def bezier_curve_derivative(self, t, control_points):
        control_points = np.array(control_points)  # 转换为 NumPy 数组
        n = len(control_points) - 1
        derivative = np.zeros(2)
        for i in range(n):
            binomial_coeff = np.math.comb(n - 1, i)
            derivative = derivative + n * binomial_coeff * (t ** i) * ((1 - t) ** (n - 1 - i)) * (control_points[i + 1] - control_points[i])
        return derivative

    def bezier_curve_kth_derivative(self, t, control_points, k):
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
    
    # 计算贝塞尔曲线的曲率
    def bezier_curve_curvature(self, t, control_points):
        
        first_derivative = self.bezier_curve_kth_derivative(t, control_points, 1)
        second_derivative = self.bezier_curve_kth_derivative(t, control_points, 2)
        # print(f"{t}  first_derivative: {first_derivative.value}")
        # print(f"{t}  second_derivative: {second_derivative.value}")
        
        # numerator = cp.abs(first_derivative[0] * second_derivative[1] - first_derivative[1] * second_derivative[0])
        # 正数为向上弯曲，负数为向下弯曲
        numerator = first_derivative[0] * second_derivative[1] - first_derivative[1] * second_derivative[0]
        denominator = cp.norm(first_derivative) ** 3
        
        epsilon = 1e-6
        curvature = numerator / (denominator + epsilon)

        return curvature

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

    def calculate_uniform_bezier_points(self, control_points):
        total_length = self.arc_length(control_points)
        uniform_arc_lengths = np.linspace(0, total_length, self.discrete_points_num)
        uniform_points = [self.bezier_curve(self.find_t_for_arc_length(control_points, s), control_points) for s in uniform_arc_lengths]
        uniform_points[0] = control_points[0]  # 确保起点等于第一个控制点
        uniform_points[-1] = control_points[-1]  # 确保终点等于最后一个控制点
        return np.array(uniform_points)
    
    def plot_regions(self, ax):
        for region in self.regions:
            center = np.array(region["center"])
            size = np.array(region["size"])
            lower_left = center - size / 2
            rect = plt.Rectangle(lower_left, size[0], size[1], edgecolor='black', facecolor='lightgray')
            ax.add_patch(rect)
            ax.text(center[0], center[1], region["name"], ha='center', va='center', fontsize=8, color='black')

            #  # 计算并标注长方形的四个端点坐标
            # lower_right = lower_left + [size[0], 0]
            # upper_left = lower_left + [0, size[1]]
            # upper_right = lower_left + size

            # for point in [lower_left, lower_right, upper_left, upper_right]:
            #     ax.text(point[0], point[1], f'({point[0]:.2f}, {point[1]:.2f})', fontsize=6, color='blue')
        
        # 设置图形属性
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Path Visualization')

    def plot_bezier_curve(self, ax, uniform_points, control_points):
        for name, points in uniform_points.items():
            ax.plot(points[:, 0], points[:, 1], 'ro', markersize=1)
        for name, points in control_points.items():
            for point in points:
                if point is not None:
                    ax.plot(point[0], point[1], 'bo')

        ax.plot(self.start_point[0], self.start_point[1], 'go')  # 起点绿色圆点
        ax.plot(self.end_point[0], self.end_point[1], 'ro')  # 终点红色圆点

    def plot_bezier_curve_curvature(self, ax, all_curvatures):
        # 绘制曲率值折线图
        ax.plot(all_curvatures, marker='o')
        ax.set_xlabel('points')
        ax.set_ylabel('curvatures')
        ax.set_title('curvatures-points')

        # 根据数据范围调整横坐标和纵坐标的最小单位
        ax.xaxis.set_major_locator(MultipleLocator(max(1, len(all_curvatures) // 7)))
        ax.yaxis.set_major_locator(MultipleLocator(max(0.1, (max(all_curvatures) - min(all_curvatures)) / 10)))


        # 在右上角显示 order 和 continuity
        ax.text(0.95, 0.95, f'order = {self.order}\ncontinuity = {self.continuity}', 
                horizontalalignment='right', verticalalignment='top', 
                transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.5))

        # # 标注每10格的path区域
        # for i, name in enumerate(self.best_path):
        #     start_index = i * self.discrete_points_num
        #     end_index = (i + 1) * self.discrete_points_num
        #     ax.axvspan(start_index, end_index, color='gray', alpha=0.3)
        #     ax.text((start_index + end_index) / 2, min(all_curvatures) - 0.3, name, 
        #             horizontalalignment='center', verticalalignment='bottom')

        ax.grid(True)

    def plot_bezier_curve_speed(self, ax, all_speed):
        # 绘制速度值折线图
        ax.plot(all_speed, marker='o')
        ax.set_xlabel('points')
        ax.set_ylabel('speed')
        ax.set_title('speed-points')

        # 根据数据范围调整横坐标和纵坐标的最小单位
        ax.xaxis.set_major_locator(MultipleLocator(max(1, len(all_speed) // 7)))
        ax.yaxis.set_major_locator(MultipleLocator(max(0.1, (max(all_speed) - min(all_speed)) / 10)))

        # # 标注每10格的path区域
        # for i, name in enumerate(self.best_path):
        #     start_index = i * self.discrete_points_num
        #     end_index = (i + 1) * self.discrete_points_num
        #     ax.axvspan(start_index, end_index, color='gray', alpha=0.3)
        #     ax.text((start_index + end_index) / 2, min(all_speed) - 0.3, name, 
        #             horizontalalignment='center', verticalalignment='bottom')

        ax.grid(True)

    def plot_bezier_curve_acceleration(self, ax, all_acceleration):
        # 绘制加速度值折线图
        ax.plot(all_acceleration, marker='o')
        ax.set_xlabel('points')
        ax.set_ylabel('accerlation')
        ax.set_title('accerlation-points')

        # 根据数据范围调整横坐标和纵坐标的最小单位
        ax.xaxis.set_major_locator(MultipleLocator(max(1, len(all_acceleration) // 7)))
        ax.yaxis.set_major_locator(MultipleLocator(max(0.1, (max(all_acceleration) - min(all_acceleration)) / 10)))

        ax.grid(True)

    def plot_bezier_curve_headings(self, ax, all_headings):
        # 绘制航向角值折线图
        ax.plot(all_headings, marker='o')
        ax.set_xlabel('points')
        ax.set_ylabel('headings')
        ax.set_title('headings-points')

        # 根据数据范围调整横坐标和纵坐标的最小单位
        ax.xaxis.set_major_locator(MultipleLocator(max(1, len(all_headings) // 7)))
        ax.yaxis.set_major_locator(MultipleLocator(max(0.1, (max(all_headings) - min(all_headings)) / 10)))

        ax.grid(True)

    def solve(self):
        path = self.best_path  # 已知路径

        # 创建变量
        points = {}
        constraints = []

        assert self.continuity < self.order

        # 初始化所有路径点
        for name in path:
            points[name] = [cp.Variable(2) for _ in range(self.order + 1)]

        # 添加约束
        for i, region in enumerate(self.regions):
            name = region["name"]
            if name in path:
                x = points[name]
                c = np.array(region["center"])
                size = np.array(region["size"]) / 2
                # 添加长方形约束
                for j in range(self.order + 1):
                    if j == 0 or j == self.order:
                        constraints.append(x[j] >= c - size)
                        constraints.append(x[j] <= c + size)
                    else:
                        # constraints.append(x[j] >= c - 0.85 * size)
                        # constraints.append(x[j] <= c + 0.85 * size)
                        constraints.append(x[j] >= c - size)
                        constraints.append(x[j] <= c + size)


                # 起点和终点约束
                if name == path[0]:
                    constraints.append(x[0] == self.start_point)
                elif name == path[-1]:
                    constraints.append(x[-1] == self.end_point)

                # 添加路径约束
                next_index = path.index(name) + 1
                if name != path[-1]:
                    next_name = path[next_index]
                    # 位置连续性
                    constraints.append(points[name][-1] == points[next_name][0])

                    # 速度连续性
                    derivative_current = self.bezier_curve_kth_derivative(1, points[name], 1)
                    derivative_next = self.bezier_curve_kth_derivative(0, points[next_name], 1)
                    constraints.append(derivative_current == derivative_next)

                    # 加速度连续性
                    second_derivative_current = self.bezier_curve_kth_derivative(1, points[name], 2)
                    second_derivative_next = self.bezier_curve_kth_derivative(0, points[next_name], 2)
                    constraints.append(second_derivative_current == second_derivative_next)

        # 目标函数：曲线总长度       
        # sum_length = 0
        # curve_points_all = {}
        # for name in path:
        #     x = points[name]
        #     t_values = np.linspace(0, 1, self.discrete_points_num)
        #     curve_points = np.array([BezierCurves.bezier_curve(t, x) for t in t_values])
        #     curve_points_all[name] = curve_points

        #     for k in range(1, len(curve_points)):
        #         x1 = curve_points[k - 1]
        #         x2 = curve_points[k]
        #         sum_length += cp.norm(x1 - x2)

        # 目标函数：控制点距离
        sum_length = 0
        for name in path:
            x = points[name]
            for i in range(1, self.order + 1):
                sum_length += cp.abs(cp.norm(x[i] - x[i - 1]))

        goal = sum_length

        objective = cp.Minimize(goal)



        # 创建并求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        
        control_points = {name: [point.value for point in points[name]] for name in path}
        # print(f"control_points: {control_points}")
        curve_points_all = {}
        for name in path:
            x = control_points[name]
            t_values = np.linspace(0, 1, self.discrete_points_num)
            curve_points = np.array([BezierCurves.bezier_curve(t, x) for t in t_values])
            curve_points_all[name] = curve_points
        # print(f"curve_points_all: {curve_points_all}")
            # if name ==path[0]:
            #     print(f"curve_points:{curve_points}")
        uniform_points = {name: self.calculate_uniform_bezier_points(control_points[name]) for name in path}

        curvatures = {}
        for name in path:
            x = control_points[name]
            t_values = np.linspace(0, 1, self.discrete_points_num)
            curvatures[name] = [self.bezier_curve_curvature(t, x).value for t in t_values]
        curvatures_all = np.concatenate(list(curvatures.values()))

        speed = {}
        for name in path:
            x = control_points[name]
            t_values = np.linspace(0, 1, self.discrete_points_num)
            speed[name] = [cp.norm(self.bezier_curve_kth_derivative(t, x, 1)).value for t in t_values]
        speed_all = np.concatenate(list(speed.values()))

        acceleration = {}
        for name in path:
            x = control_points[name]
            t_values = np.linspace(0, 1, self.discrete_points_num)
            acceleration[name] = [cp.norm(self.bezier_curve_kth_derivative(t, x, 2)).value for t in t_values]
        acceleration_all = np.concatenate(list(acceleration.values()))

        headings = {}
        for name in path:
            x = control_points[name]
            t_values = np.linspace(0, 1, self.discrete_points_num)
            dy = [self.bezier_curve_kth_derivative(t, x, 1)[1] for t in t_values]
            dx = [self.bezier_curve_kth_derivative(t, x, 1)[0] for t in t_values]
            headings[name] = np.arctan2(dy, dx)
        headings_all = np.concatenate(list(headings.values()))

        return control_points, curvatures_all, curve_points_all, speed_all, uniform_points, acceleration_all, headings_all