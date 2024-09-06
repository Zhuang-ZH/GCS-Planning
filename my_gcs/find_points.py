import numpy as np
import cvxpy as cp


def linear(best_path, regions, start_point, end_point):
    # 创建变量
    points = {}
    constraints = []

    # 初始化所有路径点
    for name in best_path:
        points[name] = (cp.Variable(2), cp.Variable(2))

    # 添加约束
    sum_length = 0
    for i, region in enumerate(regions):
        name = region["name"]
        if name in best_path:
            x1, x2 = points[name]
            c = np.array(region["center"])
            size = np.array(region["size"]) / 2
            # 添加长方形约束
            constraints.append(x1 >= c - size)
            constraints.append(x1 <= c + size)
            constraints.append(x2 >= c - size)
            constraints.append(x2 <= c + size)

            # 起点和终点约束
            if name == best_path[0]:
                constraints.append(x1 == start_point)
            elif name == best_path[-1]:
                constraints.append(x2 == end_point)

            # 添加路径约束
            next_index = best_path.index(name) + 1
            if name != best_path[-1]:
                next_name = best_path[next_index]
                constraints.append(points[name][1] == points[next_name][0])

            sum_length += cp.norm(x1 - x2)



    # 定义目标函数，最小化总距离
    objective = cp.Minimize(sum_length)

    # 创建并求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=True)


    # 输出结果
    for name in best_path:
        if isinstance(points[name], tuple):
            x1, x2 = points[name]
            print(f"Point in {name}: ({x1.value}, {x2.value})")
        else:
            print(f"Point in {name}: {points[name].value}")
        
    return points

    