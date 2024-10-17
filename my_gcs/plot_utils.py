import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv

# 生成贝塞尔曲线
def bezier_curve(control_points, points_num=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, points_num)
    curve = np.zeros((points_num, 2))
    for i in range(points_num):
        for j in range(n + 1):
            curve[i] += control_points[j] * bernstein_poly(j, n, t[i])
    return curve

# 伯恩斯坦多项式
def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

# 组合数
def comb(n, k):
    return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))

def discretize_vertex_2d(vertex, n=50, is_start=False, is_goal=False):
    values = get_values(vertex)
    y_relaxed_value = vertex.y_relaxed.value  # 保存 y_relaxed 的值
    if is_start:
        variable = vertex.variables[-1]
    elif is_goal:
        variable = vertex.variables[0]
    else:
        variable = vertex.variables[0]
    
    # print(vertex.constraints)
    
    cost = cp.Parameter(2)
    prob = cp.Problem(cp.Maximize(cost @ variable), vertex.constraints)
    vertices = np.zeros((n, 2))

    for i, angle in enumerate(np.linspace(0, 2 * np.pi, n)):
        cost.value = np.array([np.cos(angle), np.sin(angle)])
        # prob.solve() 方法会求解优化问题，并更新涉及的变量的值，包括 vertex.y_relaxed.value
        prob.solve(warm_start=True)
        vertices[i] = variable.value

    # print(f"Discretized vertices for {vertex.name}: {vertices}")

    vertex.y_relaxed.value = y_relaxed_value  # 恢复 y_relaxed 的值
    set_value(vertex, values)
    return vertices


def get_values(vertex):
    return [variable.value for variable in vertex.variables]


def set_value(vertex, values):
    for variable, value in zip(vertex.variables, values):
        variable.value = value


def plot_vertex_2d(vertex, n=50, is_start=False, is_goal=False, tol=1e-4, **kwargs):
    vertices = discretize_vertex_2d(vertex, n,  is_start, is_goal)
    
    print(f"Plotting vertex: {vertex.name}")
    for i in range(len(vertex.variables)):
        print(f"Vertex Variable {i}: {vertex.variables[i].value}")
    # print(f"Discretized vertices: {vertices}")

    options = {'facecolor': 'mintcream', 'ec': 'black', 'alpha': 0.5}
    options.update(kwargs)
    vertex_min = np.min(vertices, axis=0)
    vertex_max = np.max(vertices, axis=0)
    vertex_dist = np.linalg.norm(vertex_max - vertex_min)
    if vertex_dist <= tol:
        plt.scatter(*vertices[0], fc='k', ec='k')
    else:
        plt.fill(*vertices.T, **options, zorder=0)
        
    value = vertex.variables[0].value
    x_mean = np.mean(vertices[:, 0])
    y_mean = np.mean(vertices[:, 1])
    plt.text(x_mean, y_mean, vertex.name, fontsize=12, ha='center', va='center', zorder=1)
    

def plot_edge_2d(edge, endpoints=None, **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    for variables in [edge.tail.variables, edge.head.variables]:
        if variables[0].size != 2:
            raise ValueError("Can only plot 2D sets.")
    arrowstyle = "->, head_width=3, head_length=8"
    options = dict(zorder=2, arrowstyle=arrowstyle)
    options.update(kwargs)
    if endpoints is None:
        endpoints = closest_points(edge.tail, edge.head)
    arrow = patches.FancyArrowPatch(*endpoints, **options)
    plt.gca().add_patch(arrow)


def closest_points(vertex1, vertex2):
    y_relaxed_value_1 = vertex1.y_relaxed.value  # 保存 y_relaxed 的值
    y_relaxed_value_2 = vertex2.y_relaxed.value  # 保存 y_relaxed 的值
    values1 = get_values(vertex1)
    values2 = get_values(vertex2)
    variable1 = vertex1.variables[0]
    variable2 = vertex2.variables[0]
    cost = cp.sum_squares(variable2 - variable1)
    constraints = vertex1.constraints + vertex2.constraints
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    points = [variable1.value, variable2.value]
    set_value(vertex1, values1)
    set_value(vertex2, values2)

    vertex1.y_relaxed.value = y_relaxed_value_1  # 恢复 y_relaxed 的值
    vertex2.y_relaxed.value = y_relaxed_value_2  # 恢复 y_relaxed 的值
    return points


def plot_gcs_2d(gcs, n=50, is_plot_edges=True):
    for vertex in gcs.vertices:
        if vertex.variables[0].size != 2:
            raise ValueError("Can only plot 2D sets.")
        if vertex.name == "s":
            plot_vertex_2d(vertex, n=n, is_start = True)
        elif vertex.name == "t":
            plot_vertex_2d(vertex, n=n, is_goal = True)
        else:
            plot_vertex_2d(vertex, n=n)
    if is_plot_edges:
        for edge in gcs.edges:
            plot_edge_2d(edge, color='grey')

def plot_gcs_velocity_2d(inner_radius, outer_radius):
    def hexagon_vertices(center, radius):
        """计算六边形的顶点"""
        angles = np.linspace(0, 2 * np.pi, 7)
        return np.array([center + radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])
    outer_center = np.array([0, 0])
    inner_center = np.array([0, 0])
    # 计算大六边形和小六边形的顶点
    outer_hexagon = hexagon_vertices(outer_center, outer_radius)
    inner_hexagon = hexagon_vertices(inner_center, inner_radius)
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


def plot_subgraph_2d(gcs, tol=1e-4):
    for vertex in gcs.vertices:
        vertex_index = gcs.vertices.index(vertex)
        print(f"vertex {vertex_index}: {vertex.y_relaxed.value}")
        if vertex.y.value is not None and vertex.y.value > tol:
            variable = vertex.variables[0]
            plt.scatter(*variable.value, fc='w', ec='k', zorder=3)
    for edge in gcs.edges:
        edge_index = gcs.edges.index(edge)
        print(f"edge {edge_index}: {edge.y_relaxed.value}")
        if edge.y.value is not None and edge.y.value > tol:
            tail = edge.tail.variables[0].value
            head = edge.head.variables[0].value
            endpoints = (tail, head)
            plot_edge_2d(edge, endpoints, color='blue', linestyle='--')

def plot_relaxed_subgraph_2d(gcs, tol=1e-4):

    for i, vertex in enumerate(gcs.vertices):
        vertex_index = gcs.vertices.index(vertex)
        # print(f"vertex {vertex_index}: {vertex.y_relaxed.value}")
        if vertex.y_relaxed.value is not None and vertex.y_relaxed.value > tol:
            # for variable in vertex.variables:
                # plt.scatter(*variable.value, fc='w', ec='k', zorder=3)
            control_points = np.array([variable.value for variable in vertex.variables])
            # print(f"control_points: {control_points}")
            plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')

            curve = bezier_curve(control_points)
            if vertex.name == "s":
                plt.plot(curve[:, 0], curve[:, 1], color='red', label='Bezier Curve')
            elif vertex.name == "v1" or vertex.name == "v2":
                plt.plot(curve[:, 0], curve[:, 1], color='blue', label='Bezier Curve')
            elif vertex.name == "t":
                plt.plot(curve[:, 0], curve[:, 1], color='green', label='Bezier Curve')
            else:
                plt.plot(curve[:, 0], curve[:, 1], label='Bezier Curve')
    
    # for edge in gcs.edges:
    #     edge_index = gcs.edges.index(edge)
    #     # print(f"edge {edge_index}: {edge.y_relaxed.value}")
    #     if edge.y_relaxed.value is not None and edge.y_relaxed.value > tol:
    #         tail = edge.tail.variables[0].value
    #         head = edge.head.variables[0].value
    #         endpoints = (tail, head)
    #         plot_edge_2d(edge, endpoints, color='blue', linestyle='--')

    #         # 计算边的中点位置
    #         mid_point = [(tail[0] + head[0]) / 2, (tail[1] + head[1]) / 2]
    #         # 在中点位置添加文本标签
    #         plt.text(mid_point[0], mid_point[1], f'{edge.y_relaxed.value:.2f}', color='red', fontsize=8, ha='center')

def plot_subgraph_velocity_2d(Q0, Q1, Q2):
    bezier_curve_velocity_points_s = bezier_curve([Q0[0], Q1[0], Q2[0]])
    bezier_curve_velocity_points_v = bezier_curve([Q0[1], Q1[1], Q2[1]])
    bezier_curve_velocity_points_t = bezier_curve([Q0[2], Q1[2], Q2[2]])

    bezier_curve_velocity_points = np.vstack((bezier_curve_velocity_points_s, bezier_curve_velocity_points_v, bezier_curve_velocity_points_t))

    # 绘制速度空间中的贝塞尔曲线
    plt.plot(bezier_curve_velocity_points_s[:, 0], bezier_curve_velocity_points_s[:, 1], color='red', label='Velocity Bezier Curve S')
    plt.plot(bezier_curve_velocity_points_v[:, 0], bezier_curve_velocity_points_v[:, 1], color='blue', label='Velocity Bezier Curve V')
    plt.plot(bezier_curve_velocity_points_t[:, 0], bezier_curve_velocity_points_t[:, 1], color='green', label='Velocity Bezier Curve T')
    # 绘制控制点
    plt.scatter([Q0[0][0], Q1[0][0], Q2[0][0]], [Q0[0][1], Q1[0][1], Q2[0][1]], color='red', label='Control Points S')
    plt.scatter([Q0[1][0], Q1[1][0], Q2[1][0]], [Q0[1][1], Q1[1][1], Q2[1][1]], color='blue', label='Control Points V')
    plt.scatter([Q0[2][0], Q1[2][0], Q2[2][0]], [Q0[2][1], Q1[2][1], Q2[2][1]], color='green', label='Control Points T')

    plt.title('Velocity Space Bezier Curve')
    plt.xlabel('X Velocity')
    plt.ylabel('Y Velocity')
    plt.legend()
    plt.grid(True)

def plot_corridors(gcs, corridors, **kwargs):
    for vertex in corridors:
            plot_vertex_2d(vertex, facecolor='darkblue', alpha=0.5, **kwargs)

# def plot_path(gcs, control_points, **kwargs):
#     for points in control_points:
        

def graphviz_gcs(gcs, vertex_labels=None, edge_labels=None):
    if vertex_labels is None:
        vertex_labels = [vertex.name for vertex in gcs.vertices]
    if edge_labels is None:
        edge_labels = [''] * gcs.num_edges()
    digraph = gv.Digraph()
    for label in vertex_labels:
        digraph.node(str(label))
    for edge, label in zip(gcs.edges, edge_labels):
        tail = vertex_labels[gcs.vertices.index(edge.tail)]
        head = vertex_labels[gcs.vertices.index(edge.head)]
        digraph.edge(str(tail), str(head), str(label))
    return digraph
