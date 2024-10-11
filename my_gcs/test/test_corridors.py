isolated_corridors = [
    {'name': 'v8', 'center': [-1.75, 1.75], 'size': [0.5, 0.5]},
    {'name': 'v16', 'center': [-1.25, 1.75], 'size': [0.5, 0.5]},
    {'name': 'v24', 'center': [-0.75, 1.75], 'size': [0.5, 0.5]},
    {'name': 'v23', 'center': [-0.75, 1.25], 'size': [0.5, 0.5]},
    {'name': 'v31', 'center': [-0.25, 1.25], 'size': [0.5, 0.5]},
    {'name': 'v30', 'center': [-0.25, 0.75], 'size': [0.5, 0.5]},
    {'name': 'v38', 'center': [0.25, 0.75], 'size': [0.5, 0.5]},
    {'name': 'v46', 'center': [0.75, 0.75], 'size': [0.5, 0.5]},
    {'name': 'v45', 'center': [0.75, 0.25], 'size': [0.5, 0.5]},
    {'name': 'v44', 'center': [0.75, -0.25], 'size': [0.5, 0.5]},
    {'name': 'v52', 'center': [1.25, -0.25], 'size': [0.5, 0.5]},
    {'name': 'v51', 'center': [1.25, -0.75], 'size': [0.5, 0.5]},
    {'name': 'v50', 'center': [1.25, -1.25], 'size': [0.5, 0.5]},
    {'name': 'v49', 'center': [1.25, -1.75], 'size': [0.5, 0.5]},
    {'name': 'v57', 'center': [1.75, -1.75], 'size': [0.5, 0.5]}
]


centers = [region['center'] for region in isolated_corridors]
route = centers
print(route)

# 计算每个 isolated_corridor 的边界
corridors = []
for corridor in isolated_corridors:
    center = corridor['center']
    size = corridor['size']
    xmin = center[0] - size[0] / 2
    xmax = center[0] + size[0] / 2
    ymin = center[1] - size[1] / 2
    ymax = center[1] + size[1] / 2
    corridors.append([xmin, xmax, ymin, ymax])

# 合并相邻的 corridors
merged_corridors = []
for i in range(len(corridors) - 1):
    corridor1 = corridors[i]
    corridor2 = corridors[i + 1]
    xmin = min(corridor1[0], corridor2[0])
    xmax = max(corridor1[1], corridor2[1])
    ymin = min(corridor1[2], corridor2[2])
    ymax = max(corridor1[3], corridor2[3])
    merged_corridors.append([xmin, xmax, ymin, ymax])

print(merged_corridors)

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

print(merged_corridors)

# 输出结果
corridor_string = "; ".join([f"{corridor[0]} {corridor[1]} {corridor[2]} {corridor[3]}" for corridor in merged_corridors])
output_string = f"corridor1 = [{corridor_string}]"

# print(output_string)