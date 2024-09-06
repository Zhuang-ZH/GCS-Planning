import cvxpy as cp
import numpy as np
import sys

sys.path.append('../PathPlanning/GCS_Planning/my_gcs')
from graph_of_convex_sets import GraphOfConvexSets, Vertex
import matplotlib.pyplot as plt
from RandomDFS_relaxed import randomForwardPathSearch, findMaxWeightPath
from find_points import linear
from BezierCurves import BezierCurves
from matplotlib.ticker import MultipleLocator

# 判断障碍物是否重叠
def is_overlapping(rect1, rect2, tolerance=1e-6):
    x1_min, y1_min = rect1["center"][0] - rect1["size"][0] / 2, rect1["center"][1] - rect1["size"][1] / 2
    x1_max, y1_max = rect1["center"][0] + rect1["size"][0] / 2, rect1["center"][1] + rect1["size"][1] / 2
    x2_min, y2_min = rect2["center"][0] - rect2["size"][0] / 2, rect2["center"][1] - rect2["size"][1] / 2
    x2_max, y2_max = rect2["center"][0] + rect2["size"][0] / 2, rect2["center"][1] + rect2["size"][1] / 2

    return not (x1_max - tolerance <= x2_min or x1_min + tolerance >= x2_max or y1_max - tolerance <= y2_min or y1_min + tolerance >= y2_max)

# 判断两个矩形是否相交
def is_union(rect1, rect2, tolerance=1e-6):
    x1_min, y1_min = rect1["center"][0] - rect1["size"][0] / 2, rect1["center"][1] - rect1["size"][1] / 2
    x1_max, y1_max = rect1["center"][0] + rect1["size"][0] / 2, rect1["center"][1] + rect1["size"][1] / 2
    x2_min, y2_min = rect2["center"][0] - rect2["size"][0] / 2, rect2["center"][1] - rect2["size"][1] / 2
    x2_max, y2_max = rect2["center"][0] + rect2["size"][0] / 2, rect2["center"][1] + rect2["size"][1] / 2

    return not (x1_max + tolerance < x2_min or x1_min - tolerance > x2_max or y1_max + tolerance < y2_min or y1_min - tolerance > y2_max)

def is_covered_by_obstacles(rect, obstacles):
    for obstacle in obstacles:
        if is_overlapping(rect, obstacle):
            return True
    return False

def column_can_merge(rect1, rect2, epsilon=1e-6):
    return (abs(rect1["center"][0] - rect2["center"][0]) < epsilon and
            abs(rect1["size"][0] - rect2["size"][0]) < epsilon and
            abs(abs(rect1["center"][1] - rect2["center"][1]) - (rect1["size"][1] + rect2["size"][1]) / 2) < epsilon)

def column_merge_rects(rect1, rect2):
    new_center_y = (rect1["center"][1] + rect2["size"][1] / 2)
    new_size_y = rect1["size"][1] + rect2["size"][1]
    return {"name": rect1["name"], "center": [rect1["center"][0], new_center_y], "size": [rect1["size"][0], new_size_y]}

def row_can_merge(rect1, rect2, epsilon=1e-6):
    return (abs(rect1["center"][1] - rect2["center"][1]) < epsilon and
            abs(rect1["size"][1] - rect2["size"][1]) < epsilon and
            abs(abs(rect1["center"][0] - rect2["center"][0]) - (rect1["size"][0] + rect2["size"][0]) / 2) < epsilon)

def row_merge_rects(rect1, rect2):
    new_center_x = (rect1["center"][0] + rect2["size"][0] / 2)
    new_size_x = rect1["size"][0] + rect2["size"][0]
    return {"name": rect1["name"], "center": [new_center_x, rect1["center"][1]], "size": [new_size_x, rect1["size"][1]]}

def generate_regions(x_min, x_max, y_min, y_max, obstacles):
    regions = []
    region_width = 0.1  # 假设每个区域的宽度为0.5
    region_height = 0.1  # 假设每个区域的高度为0.5
    region_id = 1

    x = x_min
    while x < x_max:
        y = y_min
        column_regions = []
        while y < y_max:
            region = {"name": f"v{region_id}", "center": [x + region_width / 2, y + region_height / 2], "size": [region_width, region_height]}
            if not is_covered_by_obstacles(region, obstacles):
                column_regions.append(region)
            y += region_height
            region_id += 1
        
        # 合并相邻的矩形
        merged_regions = []
        i = 0
        while i < len(column_regions):
            current_region = column_regions[i]
            while i + 1 < len(column_regions) and column_can_merge(current_region, column_regions[i + 1]):
                current_region = column_merge_rects(current_region, column_regions[i + 1])   
                i += 1
            merged_regions.append(current_region)
            i += 1
 
        regions.extend(merged_regions)
        x += region_width

    # 按行合并相邻的矩形
    final_regions = []
    i = 0
    while i < len(regions):
        j = i + 1
        current_region = regions[i]
        while j < len(regions):
            if row_can_merge(current_region, regions[j]):
                current_region = row_merge_rects(current_region, regions[j])
                regions.pop(j)
            else:
                j += 1
        final_regions.append(current_region)
        i += 1

    return final_regions

def generate_edges(regions):
    edges = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if is_union(regions[i], regions[j]):
                edges.append((regions[i]["name"], regions[j]["name"]))
                # edges.append((regions[j]["name"], regions[i]["name"]))
    return edges

def main():
    # 定义obstacles
    obstacles = [
        {"name": "ob1", "center": [-1.25, -0.5], "size": [0.5, 2.0]},
        {"name": "ob2", "center": [-0.75, 1.25], "size": [0.5, 0.5]},
        {"name": "ob3", "center": [0.75, 0.5], "size": [1.5, 1.0]},
        {"name": "ob4", "center": [1.25, -0.75], "size": [0.5, 0.5]},
    ]

    # 定义范围
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2

    # 定义起点和终点
    start_point = np.array([0.5,1.5])
    end_point = np.array([2, -1.5])

    # # 生成regions
    # regions = generate_regions(x_min, x_max, y_min, y_max, obstacles)
    # # print(regions)
    # gcs = GraphOfConvexSets()
    # # 添加regions到GCS
    # vertices = gcs.add_region_to_GCS(regions)
    # # print(vertices)
    # edges = generate_edges(regions)
    # print(edges)
    # gcs.add_edge_to_GCS(edges, vertices)
    # # print(edges)
    # gcs.graphviz()

    gcs = GraphOfConvexSets()
    regions = [
    {"name": "v1", "center": [-1.75, 0.0], "size": [0.5, 4.0]},
    {"name": "v2", "center": [-1.25, 1.25], "size": [0.5, 1.5]},
    {"name": "v3", "center": [-1.25, -1.75], "size": [0.5, 0.5]},
    {"name": "v4", "center": [-0.75, 1.75], "size": [0.5, 0.5]},
    {"name": "v5", "center": [-0.75, -0.5], "size": [0.5, 3.0]},
    {"name": "v6", "center": [-0.25, 0.0], "size": [0.5, 4.0]},
    {"name": "v7", "center": [0.75, 1.5], "size": [1.5, 1.0]},
    {"name": "v8", "center": [0.5, -1.0], "size": [1.0, 2.0]},
    {"name": "v9", "center": [1.25, -0.25], "size": [0.5, 0.5]},
    {"name": "v10", "center": [1.25, -1.5], "size": [0.5, 1.0]},
    {"name": "v11", "center": [1.75, 0.0], "size": [0.5, 4.0]},
]

    vertices = gcs.add_region_to_GCS(regions)
    edges = [
        ("v1", "v3"),
        ("v1", "v2"),
        ("v2", "v4"),
        ("v2", "v5"),
        ("v3", "v5"),
        ("v4", "v6"),
        ("v5", "v6"),
        ("v6", "v7"),
        ("v6", "v8"),
        ("v8", "v9"),
        ("v8", "v10"),
        ("v7", "v11"),
        ("v9", "v11"),
        ("v10", "v11"),
    ]
    gcs.add_edge_to_GCS(edges, vertices)

    start_region, end_region = gcs.find_start_end_region(start_point, end_point, regions)
    print(f"start_region: {start_region}")
    print(f"end_region: {end_region}")

    s = vertices[start_region][0]
    t = vertices[end_region][0]

    prob = gcs.solve_shortest_path_relaxation(s, t)
    print('Problem status:', prob.status)
    print('Optimal value:', prob.value)

    gcs.plot_2d()
    plt.show()

if __name__ == "__main__":
    main()