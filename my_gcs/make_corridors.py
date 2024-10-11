def make_corridor(regions, path):
    # corridors = [vertices[name][0] for name in best_path]
    isolated_corridors = [next(region for region in regions if region["name"] == name) for name in path]
    # print(isolated_corridors)

    centers = [region['center'] for region in isolated_corridors]
    route = centers
    route_matrix = "; \n".join([f"\t{center[0]} {center[1]}" for center in centers])
    route_str = f"route1 = [\n{route_matrix}\n\t] * 1.0"
    # print(route)

    # 计算每个 isolated_corridor 的边界
    corridor_edges = []
    for corridor in isolated_corridors:
        center = corridor['center']
        size = corridor['size']
        xmin = center[0] - size[0] / 2
        xmax = center[0] + size[0] / 2
        ymin = center[1] - size[1] / 2
        ymax = center[1] + size[1] / 2
        corridor_edges.append([xmin, xmax, ymin, ymax])

    # 合并相邻的 corridors
    merged_corridors = []
    for i in range(len(corridor_edges) - 1):
        corridor1 = corridor_edges[i]
        corridor2 = corridor_edges[i + 1]
        xmin = min(corridor1[0], corridor2[0])
        xmax = max(corridor1[1], corridor2[1])
        ymin = min(corridor1[2], corridor2[2])
        ymax = max(corridor1[3], corridor2[3])
        merged_corridors.append([xmin, xmax, ymin, ymax])

    for i in range(len(route)):
        if i == 0:
            # 向右
            if route[i][0] < route[i+1][0]:
                merged_corridors[i][0] -= 1.0
            # 向左
            elif route[i][0] > route[i+1][0]:
                merged_corridors[i][1] += 1.0
            # 向上
            elif route[i][1] < route[i+1][1]:
                merged_corridors[i][2] -= 1.0
            # 向下
            elif route[i][1] > route[i+1][1]:
                merged_corridors[i][3] += 1.0
        elif i == len(route) - 1:
            # 向右
            if route[i-1][0] < route[i][0]:
                merged_corridors[i-1][1] += 1.0
            # 向左
            elif route[i-1][0] > route[i][0]:
                merged_corridors[i-1][0] -= 1.0
            # 向上
            elif route[i-1][1] < route[i][1]:
                merged_corridors[i-1][3] += 1.0
            # 向下
            elif route[i-1][1] > route[i][1]:
                merged_corridors[i-1][2] -= 1.0


    # 输出结果
    corridor_string = "; \n".join([f"\t{corridor[0]} {corridor[1]} {corridor[2]} {corridor[3]}" for corridor in merged_corridors])
    corridors_str = f"corridor1 = [\n{corridor_string}\n\t] * 1.0"

    return route_str, corridors_str