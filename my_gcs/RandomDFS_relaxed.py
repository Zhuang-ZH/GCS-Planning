import numpy as np
import networkx as nx


def depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn):
    """深度优先搜索算法"""
    visited_vertices = [source]
    path_vertices = [source]
    path_edges = []
    while path_vertices[-1] != target:
        candidate_edges = getCandidateEdgesFn(path_vertices[-1], visited_vertices)
        if len(candidate_edges) == 0:
            path_vertices.pop()
            path_edges.pop()
        else:
            next_edge, next_vertex = edgeSelectorFn(candidate_edges)
            visited_vertices.append(next_vertex)
            path_vertices.append(next_vertex)
            path_edges.append(next_edge)
    return path_edges

def outgoingEdges(G):
    """获取图中每个顶点的出边"""
    outgoing_edges = {u: [] for u in G.nodes}
    for u, v, data in G.edges(data=True):
        outgoing_edges[u].append((u, v, data))
    return outgoing_edges

def extractEdgeFlows(G, result):
    """提取图中每条边的流量"""
    flows = {}
    for u, v, data in G.edges(data=True):
        edge_id = (u, v)
        flows[edge_id] = result.get(edge_id, 0)
    return flows

def randomEdgeSelector(candidate_edges, flows):
    candidate_flows = np.array([flows[(u, v)] for u, v, data in candidate_edges])
    probabilities = candidate_flows / sum(candidate_flows)
    selected_edge = np.random.choice(len(candidate_edges), p=probabilities)
    return candidate_edges[selected_edge]

# Rounding Strategies
def runTrials(source, target, getCandidateEdgesFn, edgeSelectorFn, max_trials=1000):
    paths = []
    trials = 0
    while trials < max_trials:
        trials += 1
        path = depthFirst(source, target, getCandidateEdgesFn, edgeSelectorFn)
        if path not in paths:
            paths.append(path)
    return paths

def randomForwardPathSearch(regions, edges, edge_weights, source, target, max_trials=100, seed=None, flow_tol=1e-5, **kwargs):
    """随机前向路径搜索"""

    if seed is not None:
        np.random.seed(seed)

    G = nx.DiGraph()
    for region in regions:
        G.add_node(region["name"])
    for edge in edges:
        weight = edge_weights.get(edge, 1)
        G.add_edge(edge[0], edge[1], weight=weight)

    outgoing_edges = outgoingEdges(G)
    flows = extractEdgeFlows(G, edge_weights)

    def getCandidateEdgesFn(current_vertex, visited_vertices):
        keepEdge = lambda e: e[1] not in visited_vertices and flows[(e[0], e[1])] > flow_tol
        return [e for e in outgoing_edges[current_vertex] if keepEdge(e)]

    def edgeSelectorFn(candidate_edges):
        e = randomEdgeSelector(candidate_edges, flows)
        return e, e[1]

    return runTrials(source, target, getCandidateEdgesFn, edgeSelectorFn, max_trials)

def findMinCostPath(paths, cost):
    min_cost = float('inf')
    min_cost_path = None
    for path in paths:
        total_cost = sum(cost.get((edge[0], edge[1]), float('inf')) for edge in path)
        if total_cost < min_cost:
            min_cost = total_cost
            min_cost_path = path

    # 将边列表转换为顶点列表
    if min_cost_path:
        vertex_path = [min_cost_path[0][0]]  # 添加起点
        for edge in min_cost_path:
            vertex_path.append(edge[1])  # 添加终点
        return vertex_path
    return None