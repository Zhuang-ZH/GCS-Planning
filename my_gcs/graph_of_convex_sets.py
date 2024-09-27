import cvxpy as cp
import numpy as np
from collections.abc import Iterable
from programs import ConvexProgram, ConicProgram
from graph_problems import graph_problem, shortest_path, graph_problem_relaxation, shortest_path_relaxation

class Vertex(ConvexProgram):

    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def _verify_variables(self, variables):
        ids0 = {variable.id for variable in self.variables}
        ids1 = {variable.id for variable in variables}
        if not ids0 >= ids1:
            raise ValueError("A variable does not belong to this vertex.")

    def get_feasible_point(self):
        values = [variable.value for variable in self.variables]
        prob = cp.Problem(cp.Minimize(0), self.constraints)
        prob.solve()
        feasible_point = [variable.value for variable in self.variables]
        for variable, value in zip(self.variables, values):
            variable.value = value
        return feasible_point
    

    
class Edge(ConvexProgram):

    def __init__(self, tail, head):
        super().__init__()
        self.tail = tail
        self.head = head
        self.conic_program = None
        self.name = (self.tail.name, self.head.name)

    def _verify_variables(self, variables):
        edge_variables = self.variables + self.tail.variables + self.head.variables
        ids0 = {variable.id for variable in edge_variables}
        ids1 = {variable.id for variable in variables}
        if not ids0 >= ids1:
            raise ValueError("A variable does not belong to this edge.")


class GraphOfConvexSets:

    def __init__(self):
        self.vertices = []
        self.edges = []

    def addSourceTarget(self, source_name, target_name):
        self.source = self.get_vertex_by_name(source_name)
        self.target = self.get_vertex_by_name(target_name)
        
    def add_vertex(self, name="",set = None):
        vertex = Vertex(name)
        self.vertices.append(vertex)
        return vertex
    
    def add_edge(self, tail, head):
        edge = Edge(tail, head)
        self.edges.append(edge)
        return edge

    def add_subgraph(self, gcs):
        self.vertices += gcs.vertices
        self.edges += gcs.edges

    def get_edge(self, tail, head):
        for edge in self.edges:
            if edge.tail == tail and edge.head == head:
                return edge
            
    def neighbors(self, vertex):
        neighbors = []
        for edge in self.edges:
            if edge.tail == vertex:
                neighbors.append(edge.head)
        return neighbors

    def get_vertex_by_name(self, name):
        for vertex in self.vertices:
            if vertex.name == name:
                return vertex
        raise ValueError(f"There is no vertex named {name}.")

    def get_edge_by_name(self, tail_name, head_name):
        for edge in self.edges:
            if edge.tail.name == tail_name and edge.head.name == head_name:
                return edge
        raise ValueError(f"There is no edge with tail named {tail_name} and head named {head_name}.")

    def vertex_index(self, vertex):
        return self.vertices.index(vertex)

    def edge_index(self, edge):
        return self.edges.index(edge)

    def incoming_edges(self, v):
        if isinstance(v, Vertex):
            return [e for e in self.edges if e.head == v]
        if isinstance(v, Iterable):
            return [e for e in self.edges if e.head in v and e.tail not in v]

    def outgoing_edges(self, v):
        if isinstance(v, Vertex):
            return [e for e in self.edges if e.tail == v]
        if isinstance(v, Iterable):
            return [e for e in self.edges if e.tail in v and e.head not in v]

    def incident_edges(self, v):
        return self.incoming_edges(v) + self.outgoing_edges(v)

    def incoming_indices(self, v):
        if isinstance(v, Vertex):
            return [k for k, e in enumerate(self.edges) if e.head == v]
        if isinstance(v, Iterable):
            return [k for k, e in enumerate(self.edges) if e.head in v and e.tail not in v]

    def outgoing_indices(self, v):
        if isinstance(v, Vertex):
            return [k for k, e in enumerate(self.edges) if e.tail == v]
        if isinstance(v, Iterable):
            return [k for k, e in enumerate(self.edges) if e.tail in v and e.head not in v]

    def incident_indices(self, v):
        return self.incoming_indices(v) + self.outgoing_indices(v)

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return len(self.edges)

    def vertex_binaries(self):
        return np.array([vertex.y for vertex in self.vertices])

    def edge_binaries(self):
        return np.array([edge.y for edge in self.edges])
    
    def vertex_relaxed_binaries(self):
        return np.array([vertex.y_relaxed for vertex in self.vertices])
    
    def edge_relaxed_binaries(self):
        return np.array([edge.y_relaxed for edge in self.edges])

    def to_conic(self):
        for vertex in self.vertices:
            vertex.to_conic()
        for edge in self.edges:
            edge.to_conic()

    def solve_shortest_path(self, source, target):
        problem = lambda *args: shortest_path(*args, s=source, t=target)
        return graph_problem(self, problem)
    
    def solve_shortest_path_relaxation(self, source, target):
        problem = lambda *args: shortest_path_relaxation(*args, s=source, t=target)
        return graph_problem_relaxation(self, problem)

    def solve_convex_restriction(self, vertex_indices, edge_indices):
        for k in edge_indices:
            edge = self.edges[k]
            i = self.vertex_index(edge.tail)
            j = self.vertex_index(edge.head)
            if i not in vertex_indices or j not in vertex_indices:
                raise ValueError('Given indices do not form a subgraph.')
        cost = 0
        constraints = []
        for i in vertex_indices:
            vertex = self.vertices[i]
            cost += vertex.cost
            constraints.extend(vertex.constraints)
        for k in edge_indices:
            edge = self.edges[k]
            cost += edge.cost
            constraints.extend(edge.constraints)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        for i, vertex in enumerate(self.vertices):
            vertex.y.value = 1 if i in vertex_indices else None
        for k, edge in enumerate(self.edges):
            edge.y.value = 1 if k in edge_indices else None
        return prob

    def graphviz(self):
        from plot_utils import graphviz_gcs
        return graphviz_gcs(self)

    def plot_2d(self, **kwargs):
        from plot_utils import plot_gcs_2d
        return plot_gcs_2d(self, **kwargs)

    def plot_subgraph_2d(self):
        from plot_utils import plot_subgraph_2d
        return plot_subgraph_2d(self) 
    
    def plot_relaxed_subgraph_2d(self):
        from plot_utils import plot_relaxed_subgraph_2d
        return plot_relaxed_subgraph_2d(self) 
    
    def plot_corridors(self, corridors):
        from plot_utils import plot_corridors
        return plot_corridors(self, corridors)
    
    ###
    def add_region_to_GCS(self, regions):
        vertices = {}
        for region in regions:
            v = self.add_vertex(region["name"])
            x = v.add_variable(2)
            c = np.array(region["center"])
            size = np.array(region["size"]) / 2
            # 添加长方形约束
            constraint = cp.abs(x - c) <= size
            v.add_constraint(constraint)
            vertices[region["name"]] = (v, x)
        return vertices
    
    def add_edge_to_GCS(self, edges, vertices):
        # 添加edges到GCS
        for (start, end) in edges:
            v_start, x_start = vertices[start]
            v_end, x_end = vertices[end]
            edge = self.add_edge(v_start, v_end)
            edge.add_cost(cp.norm(x_start - x_end, 2))
            # bug所在
            # edge.add_constraint(x_end[1] >= x_start[1])

    
    def find_start_end_region(self, start_point, end_point, regions):
        for region in regions:
            if start_point[0] >= region["center"][0] - region["size"][0] / 2 and start_point[0] <= region["center"][0] + region["size"][0] / 2 and start_point[1] >= region["center"][1] - region["size"][1] / 2 and start_point[1] <= region["center"][1] + region["size"][1] / 2:
                start_region = region["name"]
            if end_point[0] >= region["center"][0] - region["size"][0] / 2 and end_point[0] <= region["center"][0] + region["size"][0] / 2 and end_point[1] >= region["center"][1] - region["size"][1] / 2 and end_point[1] <= region["center"][1] + region["size"][1] / 2:
                end_region = region["name"]
        return start_region, end_region
    
    def get_edge_weights(self, relaxed=True):
        if relaxed:
            return {(edge.tail.name, edge.head.name): edge.y_relaxed.value for edge in self.edges}
        else:
            return {(edge.tail.name, edge.head.name): edge.y.value for edge in self.edges}
        
    def get_edge_costs(self):
        return {(edge.tail.name, edge.head.name): edge.cost.value for edge in self.edges}