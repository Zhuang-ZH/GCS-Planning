import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv


def discretize_vertex_2d(vertex, n=50):
    values = get_values(vertex)
    y_relaxed_value = vertex.y_relaxed.value  # 保存 y_relaxed 的值
    variable = vertex.variables[0]
    cost = cp.Parameter(2)
    prob = cp.Problem(cp.Maximize(cost @ variable), vertex.constraints)
    vertices = np.zeros((n, 2))

    for i, angle in enumerate(np.linspace(0, 2 * np.pi, n)):
        cost.value = np.array([np.cos(angle), np.sin(angle)])
        # prob.solve() 方法会求解优化问题，并更新涉及的变量的值，包括 vertex.y_relaxed.value
        prob.solve(warm_start=True)
        vertices[i] = variable.value

    vertex.y_relaxed.value = y_relaxed_value  # 恢复 y_relaxed 的值
    set_value(vertex, values)
    return vertices


def get_values(vertex):
    return [variable.value for variable in vertex.variables]


def set_value(vertex, values):
    for variable, value in zip(vertex.variables, values):
        variable.value = value


def plot_vertex_2d(vertex, n=50, tol=1e-4, **kwargs):
    vertices = discretize_vertex_2d(vertex, n)

    # print(f"Before plotting: {vertex.y_relaxed.value}")

    options = {'fc': 'mintcream', 'ec': 'black'}
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


def plot_gcs_2d(gcs, n=50):
    for vertex in gcs.vertices:
        if vertex.variables[0].size != 2:
            raise ValueError("Can only plot 2D sets.")
        plot_vertex_2d(vertex, n)
    for edge in gcs.edges:
        plot_edge_2d(edge, color='grey')



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
    for vertex in gcs.vertices:
        vertex_index = gcs.vertices.index(vertex)
        # print(f"vertex {vertex_index}: {vertex.y_relaxed.value}")
        if vertex.y_relaxed.value is not None and vertex.y_relaxed.value > tol:
            variable = vertex.variables[0]
            plt.scatter(*variable.value, fc='w', ec='k', zorder=3)
    for edge in gcs.edges:
        edge_index = gcs.edges.index(edge)
        # print(f"edge {edge_index}: {edge.y_relaxed.value}")
        if edge.y_relaxed.value is not None and edge.y_relaxed.value > tol:
            tail = edge.tail.variables[0].value
            head = edge.head.variables[0].value
            endpoints = (tail, head)
            plot_edge_2d(edge, endpoints, color='blue', linestyle='--')

            # 计算边的中点位置
            mid_point = [(tail[0] + head[0]) / 2, (tail[1] + head[1]) / 2]
            # 在中点位置添加文本标签
            plt.text(mid_point[0], mid_point[1], f'{edge.y_relaxed.value:.2f}', color='red', fontsize=8, ha='center')



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
