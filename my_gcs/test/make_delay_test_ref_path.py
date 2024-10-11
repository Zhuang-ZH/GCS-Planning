import numpy as np
import matplotlib.pyplot as plt
import csv



# 参数设定
radius1 = 2.0
radius2 = 1.0
v = 0.5  # 统一速度
delta = 0.0  # 方向盘转角

# 生成第一个半圆路径：从 (0, -1.5) 到 (0, 0)
# theta1 = np.linspace(-np.pi / 2, np.pi / 2, 500)  # 第一个半圆的角度范围
theta1 = np.linspace(np.pi / 4, np.pi / 2, 500)  # 第一个半圆的角度范围
x1 = radius1 * np.cos(theta1)
y1 = radius1 * np.sin(theta1) - 2.0  # y坐标需要平移 -1.5

# 生成第二个半圆路径：从 (0, 0) 到 (0, 1.5)，反方向
# theta2 = np.linspace(3 * np.pi / 2, np.pi / 2, 500)  # 第二个半圆的角度范围
theta2 = np.linspace(3 * np.pi / 2, np.pi, 500)  # 第二个半圆的角度范围
x2 = radius2 * np.cos(theta2)
y2 = radius2 * np.sin(theta2) + 1.0  # y坐标需要平移 +1.5

# 合并两个半圆路径
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))

# 计算偏航角 ψ（根据切线角度）
psi = np.arctan2(np.diff(y, append=y[-1]), np.diff(x, append=x[-1]))

# 计算曲率 κ（曲率公式：κ = dψ / ds, 这里用有限差分近似）
ds = np.sqrt(np.diff(x, append=x[-1])**2 + np.diff(y, append=y[-1])**2)  # 距离差
dpsi = np.diff(psi, append=psi[-1])  # 角度差

# 保证 dpsi 和 ds 的长度相同
curvature = dpsi / ds
curvature[-2] = curvature[-3]

# 将路径数据保存到CSV文件
with open('test_delay_path.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['x', 'y', 'ψ', 'v', 'κ', 'δ'])
    for i in range(len(x) - 1):
        csvwriter.writerow([x[i], y[i], psi[i], v, curvature[i], delta])

# 保存路径图到文件
plt.plot(x, y, label="S-shaped Path")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('S-shaped Path with Two Semi-circles')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
