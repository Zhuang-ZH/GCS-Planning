import cvxpy as cp
import numpy as np
from utils.maze import Maze
from my_gcs import GraphOfConvexSets

import matplotlib.pyplot as plt

maze_size = 20
knock_downs = 10

maze = Maze(maze_size, maze_size)
maze.make_maze()
maze.knock_down_walls(knock_downs)

gcs = GraphOfConvexSets()

start = [0.5, 0]
goal = [maze_size - 0.5, maze_size]
for i in range(maze_size):
    for j in range(maze_size):
        v = gcs.add_vertex(f"v{(i, j)}")
        x1 = v.add_variable(2)
        x2 = v.add_variable(2)
        v.add_cost(cp.norm(x2 - x1, 2))
        if i == 0 and j == 0:
            v.add_constraint(x1 == start)
        else:
            v.add_constraint(x1 >= [i, j])
            v.add_constraint(x1 <= [i + 1, j + 1])
        if i == maze_size - 1 and j == maze_size - 1:
            v.add_constraint(x2 == goal)
        else:
            v.add_constraint(x2 >= [i, j])
            v.add_constraint(x2 <= [i + 1, j + 1])
        
for i in range(maze_size):
    for j in range(maze_size):
        cell = maze.get_cell(i, j)
        v = gcs.get_vertex_by_name(f"v{(i, j)}")
        for direction, (di, dj) in maze.directions.items():
            if not cell.walls[direction]:
                name = f"v{(i + di, j + dj)}"
                w = gcs.get_vertex_by_name(name)
                e = gcs.add_edge(v, w)
                e.add_constraint(v.variables[1] == w.variables[0])

s = gcs.get_vertex_by_name(f"v{(0, 0)}")
t = gcs.get_vertex_by_name(f"v{(maze_size - 1, maze_size - 1)}")
prob = gcs.solve_shortest_path(s, t)
print('Problem status:', prob.status)
print('Optimal value:', prob.value)                

plt.figure()
maze.plot()
for vertex in gcs.vertices:
    if vertex.y.value > 0.5:
        x1, x2 = vertex.variables
        values = np.array([x1.value, x2.value]).T
        plt.plot(*values, c='b', linestyle='--')
plt.show()
# plt.savefig('maze.pdf')