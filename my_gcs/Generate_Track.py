import numpy as np
import matplotlib.pyplot as plt
from BezierCurves import BezierCurves
import csv

def min_vertical_distance_to_obstacles(point, obstacles):
    up_min_distance = float('inf')
    down_min_distance = float('inf')
    for obstacle in obstacles:
        center = np.array(obstacle["center"], dtype=float)  # 确保 center 是数值类型的数组
        size = np.array(obstacle["size"], dtype=float)
        lower_left = center - size / 2
        upper_right = center + size / 2

        if lower_left[1] > point[1] and upper_right[0] > point[0] and lower_left[0] < point[0]:
            distance = lower_left[1] - point[1]
            up_min_distance = min(up_min_distance, distance)
        
        if upper_right[1] < point[1] and upper_right[0] > point[0] and lower_left[0] < point[0]:
            distance = point[1] - upper_right[1]
            down_min_distance = min(down_min_distance, distance)
            
    return up_min_distance, down_min_distance

def generate_track_points(curve_points, obstacles):
    up_min_distance_global = float('inf')
    down_min_distance_global = float('inf')
    
    # 找到与障碍物下边界距离最近的点
    for key, points in curve_points.items():
        for point in points:
            up_min_distance, down_min_distance = min_vertical_distance_to_obstacles(point, obstacles)
            if up_min_distance < up_min_distance_global:
                up_min_distance_global = up_min_distance
            if down_min_distance < down_min_distance_global:
                down_min_distance_global = down_min_distance
    
    return curve_points, up_min_distance_global, down_min_distance_global

def generate_track_control_points(control_points, obstacles):
    curve_points_dict = {}
    for i, curve in enumerate(control_points):
        t_values = np.linspace(0, 1, 10)
        curve_points = np.array([BezierCurves.bezier_curve(t, control_points[curve]) for t in t_values])
        curve_points_dict[curve] = curve_points

    new_curve_points, up_min_distance_global, down_min_distance_global= generate_track_points(curve_points_dict, obstacles)
    print(f"up_min_distance_global: {up_min_distance_global}")
    print(f"down_min_distance_global: {down_min_distance_global}")
        
    up_control_points = control_points.copy()
    down_control_points = control_points.copy()
    for curve in control_points:
        # print(f"curve: {up_control_points[curve]}")
        # 将 up_control_points[curve] 转换为 numpy 数组
        up_control_points[curve] = np.array(up_control_points[curve])
        down_control_points[curve] = np.array(down_control_points[curve])
        # 在 y 向量上加上 min_distance
        up_control_points[curve][:, 1] += (up_min_distance_global - 0.05)  # 为了避免点与障碍物下边缘重合，将点向下移动 0.05
        down_control_points[curve][:, 1] -= (down_min_distance_global - 0.05)  # 为了避免点与障碍物上边缘重合，将点向上移动 0.05
    
    return up_control_points, down_control_points

def calculate_tangent(point, curve_points):
    # 计算切线方向
    idx = np.argmin(np.linalg.norm(curve_points - point, axis=1))
    if idx == 0:
        tangent = curve_points[idx + 1] - curve_points[idx]
    elif idx == len(curve_points) - 1:
        tangent = curve_points[idx] - curve_points[idx - 1]
    else:
        tangent = curve_points[idx + 1] - curve_points[idx - 1]
    return tangent / np.linalg.norm(tangent)

def calculate_normal(tangent):
    # 计算法向量（垂直于切线方向）
    return np.array([-tangent[1], tangent[0]]), np.array([tangent[1], -tangent[0]])

def calculate_perpendicular_distances(curve_points, upper_track, lower_track):
    w_tr_left_m = []
    w_tr_right_m = []

    for point in curve_points:
        # 计算切线方向
        tangent = calculate_tangent(point, curve_points)
        # 计算法向量
        normal_left, normal_right = calculate_normal(tangent)
        # 计算垂线与上轨道的交点
        upper_intersection = find_intersection_with_track(point, upper_track, normal_left)
        # 计算垂线与下轨道的交点
        lower_intersection = find_intersection_with_track(point, lower_track, normal_right)
        
        if upper_intersection is None or lower_intersection is None:
            # 如果某一侧没有交点，则w_tr_left_m=w_tr_right_m
            distance = np.linalg.norm(point - (upper_intersection if upper_intersection is not None else lower_intersection))
            w_tr_left_m.append(distance)
            w_tr_right_m.append(distance)
        else:
            # 计算垂线与上轨道交点的线段长度
            w_tr_left_m.append(np.linalg.norm(point - upper_intersection))
            # 计算垂线与下轨道交点的线段长度
            w_tr_right_m.append(np.linalg.norm(point - lower_intersection))
    
    return w_tr_left_m, w_tr_right_m

def find_intersection_with_track(point, track, normal_vector):
    
    # 计算垂线与轨道的交点
    intersections = []
    for i in range(len(track) - 1):
        segment_start = track[i]
        segment_end = track[i + 1]
        intersection = find_intersection([point, normal_vector], [segment_start, segment_end])
        if intersection is not None:
            intersections.append(intersection)
    
    if not intersections:
        return None
    
    # 找到距离最近的交点
    distances = [np.linalg.norm(point - inter) for inter in intersections]
    min_distance_idx = np.argmin(distances)
    
    return intersections[min_distance_idx]


def find_intersection(line1, line2):
    # 计算两条线段的交点
    p1, p2 = line1
    p3, p4 = line2
    denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    if denom == 0:
        return None  # 平行或重合，无交点
    x = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denom
    y = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denom
    return np.array([x, y])


def main():
    control_points = {'v1': [[-1.89924893e+00,  7.51065967e-04],
       [-1.91159764e+00, -5.10349819e-02],
       [-1.71776944e+00,  5.70751066e-01],
       [-1.59924893e+00,  7.00751066e-01]], 
       'v188': [[-1.59924893,  0.70075107],
       [-1.48072843,  0.83075107],
       [-1.2093918 ,  0.83075107],
       [-1.09924893,  0.77575108]], 
       'v388': [[-1.09924893,  0.77575108],
       [-0.98910607,  0.7207511 ],
       [-1.03383838,  0.79075107],
       [-0.89924893,  0.70075107]], 
       'v441': [[-0.89924893,  0.70075107],
       [-0.76465949,  0.61075106],
       [-0.44924944,  0.17548537],
       [-0.39924893,  0.12558149]], 
       'v641': [[-0.39924893,  0.12558149],
       [-0.34924843,  0.07567761],
       [-0.37924893,  0.09075107],
       [-0.19924893, -0.09924893]], 
       'v721': [[-0.19924893, -0.09924893],
       [-0.01924894, -0.28924893],
       [ 0.79075105, -0.28924894],
       [ 0.90075107, -0.27911429]], 
       'v1177': [[ 0.90075107, -0.27911429],
       [ 1.01075108, -0.26897965],
       [ 1.53075107, -0.2562324 ],
       [ 1.60075107, -0.27274067]], 
       'v1441': [[ 1.60075107, -0.27274067],
       [ 1.67075107, -0.28924893],
       [ 1.61075107, -0.28924893],
       [ 1.70075107, -0.20497687]], 
       'v1481': [[ 1.70075107e+00, -2.04976872e-01],
       [ 1.79075106e+00, -1.20704809e-01],
       [ 1.89123724e+00, -7.36265977e-03],
       [ 1.90075107e+00,  7.51065967e-04]]}
    
    obstacles = [
    {"name": "ob1", "center": [-1.25, -0.5], "size": [0.5, 2.0]},
    {"name": "ob2", "center": [-0.75, 1.25], "size": [0.5, 0.5]},
    {"name": "ob3", "center": [0.75, 0.5], "size": [1.5, 1.0]},
    {"name": "ob4", "center": [1.25, -0.75], "size": [0.5, 0.5]},
]

    up_control_points, down_control_points = generate_track_control_points(control_points, obstacles)
    up_bezier_points = BezierCurves.calculate_bezier_points(up_control_points)
    down_bezier_points = BezierCurves.calculate_bezier_points(down_control_points)
    reference_path_bezier_points = BezierCurves.calculate_bezier_points(control_points)

    w_tr_left_m, w_tr_right_m = calculate_perpendicular_distances(reference_path_bezier_points, up_bezier_points, down_bezier_points)

    print("w_tr_left_m_list:", w_tr_left_m)
    print("w_tr_right_m_list:", w_tr_right_m)

    # 可视化贝塞尔曲线
    plt.figure(figsize=(10, 6))
    plt.plot(up_bezier_points[:, 0], up_bezier_points[:, 1], label='Up Bezier Curve', color='blue')
    plt.plot(down_bezier_points[:, 0], down_bezier_points[:, 1], label='Down Bezier Curve', color='red')
    plt.plot(reference_path_bezier_points[:, 0], reference_path_bezier_points[:, 1], label='Reference Path Bezier Curve', color='green')

    # 添加标签和图例
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bezier Curves Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

    output_file = 'output.csv'
    # 写入新的 CSV 文件
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
        for i, point in enumerate(reference_path_bezier_points):
            writer.writerow([point[0], point[1], w_tr_left_m[i], w_tr_right_m[i]])


if __name__ == "__main__":
    main()

    