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


    return regions

def generate_edges(regions):
    edges = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if is_union(regions[i], regions[j]):
                edges.append((regions[i]["name"], regions[j]["name"]))
    return edges

def main():
    # 定义obstacles
    obstacles = [
        {"name": "ob1", "center": [-1.25, -0.5], "size": [0.5, 2.0]},
        {"name": "ob2", "center": [-0.75, 1.25], "size": [0.5, 0.5]},
        {"name": "ob3", "center": [0.75, 0.5], "size": [1.5, 1.0]},
        {"name": "ob4", "center": [1.25, -0.75], "size": [0.5, 0.5]},
    ]

    # 定义膨胀比例
    scale_factor = 1.2

    # 创建一个新的列表来存储膨胀后的障碍物
    expansion_obstacles = []

    # 按比例膨胀障碍物并存储到新的列表中
    for obstacle in obstacles:
        expanded_obstacle = {
            "name": obstacle["name"],
            "center": obstacle["center"],
            "size": [dimension * scale_factor for dimension in obstacle["size"]]
        }
        expansion_obstacles.append(expanded_obstacle)


    # 定义范围
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2

    # 生成regions
    regions = generate_regions(x_min, x_max, y_min, y_max, expansion_obstacles)
    print(regions)
    gcs = GraphOfConvexSets()
    # 添加regions到GCS
    vertices = gcs.add_region_to_GCS(regions)
    edges = generate_edges(regions)
    gcs.add_edge_to_GCS(edges, vertices)
    # print(edges)
    gcs.graphviz()
    gcs.plot_2d()
    plt.show()

if __name__ == "__main__":
    main()