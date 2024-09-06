import random
import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def random_point_in_region(region):
    # 假设 region 是一个表示凸区域的对象，并且有一个方法可以生成随机点
    # 这里我们简单地假设 region 是一个二维的矩形区域，返回随机点
    x_min, x_max, y_min, y_max = region
    return np.array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])

def average_point_in_region(region):
    x_min, x_max, y_min, y_max = region
    return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])


def dfs(graph, start, end):
    visited = set()
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        if vertex not in visited:
            if vertex == end:
                return path
            visited.add(vertex)
            neighbors = list(graph.neighbors(vertex))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                stack.append((neighbor, path + [neighbor]))
    return None

def random_dfs(start_point, end_point, edges, regions, iterations=100):
    cost_all = float('inf')
    path = []
    for i in range(iterations):
        random_points = []
        for region in regions.values():
            random_points.append(average_point_in_region(region))
        # 转换为字典
        random_points_dict = {f"v{i+1}": tuple(point) for i, point in enumerate(random_points)}

        G = nx.Graph()
        # 添加初始节点
        G.add_node("v1", pos=start_point)
        # 添加中间节点
        for i, point in enumerate(random_points[1:-1], start=2):
            G.add_node(f"v{i}", pos=point)
        # 添加最后一个节点
        G.add_node(f"v{len(random_points)}", pos=end_point)
        G.add_edges_from(edges)

        print(G.nodes(data=True))


        # # 获取节点位置
        # pos = nx.get_node_attributes(G, 'pos')

        # # 绘制图
        # plt.figure(figsize=(10, 8))
        # nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
        # plt.title("Graph Visualization")

        
        path = dfs(G, "v1", f"v{len(random_points)}")

        for points in path:
            print(points)
        cost = path_cost(random_points_dict, path)
        if cost < cost_all:
            cost_all = cost
            best_path = path
            
        print("Path from start to end in random DFS order:", path)
        print("Cost:", cost)
    # plt.show()

    print("Best cost:", cost_all)
    print("Best path:", best_path)
    return best_path, cost_all



def path_cost(random_points_dict, path):
    path_all = []
    for points in path:
        if points in random_points_dict.keys():
            path_all.append(random_points_dict[points])
    total_cost = 0.0
    for i in range(len(path) - 1):
        point1 = np.array(path_all[i])  # 转换为 numpy 数组
        point2 = np.array(path_all[i + 1])  # 转换为 numpy 数组
        # 计算两点距离
        cost = np.linalg.norm(point1 - point2)
        total_cost += cost
    return total_cost

cost1to2 = np.linalg.norm(np.array([-1.9, 0]) - np.array([-1.25,1.25]))
cost1to3 = np.linalg.norm(np.array([-1.9, 0]) - np.array([-1.25,-1.75]))
cost2to5 = np.linalg.norm(np.array([-1.25,1.25]) - np.array([-0.75, -0.5]))
cost3to5 = np.linalg.norm(np.array([-1.25,-1.75]) - np.array([-0.75,-0.5]))
print (cost1to2, cost1to3, cost2to5, cost3to5)
print (cost1to2 + cost2to5, cost1to3 + cost3to5)