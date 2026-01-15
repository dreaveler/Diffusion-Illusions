import numpy as np
from PIL import Image
from scipy.spatial import Voronoi
import math
import random

# =========================
# 参数
# =========================
IMG_SIZE = 255
NUM_CELLS = 20
JITTER = 0.08
ROTATE_RANGE = math.pi
SCALE_RANGE = (0.6, 1.3)

random.seed(1)
np.random.seed(1)

# =========================
# 生成 Voronoi 种子
# =========================
points = np.random.rand(NUM_CELLS, 2)

vor = Voronoi(points)

# 为每个 cell 定义一个局部 UV 变换
cell_params = []
for i in range(NUM_CELLS):
    cell_params.append({
        "angle": random.uniform(-ROTATE_RANGE, ROTATE_RANGE),
        "scale": random.uniform(*SCALE_RANGE),
        "offset": np.random.uniform(-JITTER, JITTER, size=2)
    })

# =========================
# 工具函数
# =========================
def rotate(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])

def nearest_site(p):
    d = np.sum((points - p) ** 2, axis=1)
    return np.argmin(d)

# =========================
# 主映射
# =========================
uv_map = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

for y in range(IMG_SIZE):
    for x in range(IMG_SIZE):
        p = np.array([x / (IMG_SIZE - 1), y / (IMG_SIZE - 1)])

        # 1️⃣ Voronoi 归属
        idx = nearest_site(p)
        site = points[idx]
        prm = cell_params[idx]

        # 2️⃣ 局部坐标
        local = p - site
        local = rotate(local, -prm["angle"])
        local /= prm["scale"]

        # 3️⃣ 映射回 UV
        uv = site + local + prm["offset"]

        # 4️⃣ 保证 UV 合法（填补而不是黑洞）
        uv = np.clip(uv, 0.0, 1.0)

        uv_map[y, x, 0] = uv[0]
        uv_map[y, x, 1] = uv[1]
        uv_map[y, x, 2] = 0.0

# =========================
# 保存
# =========================
img = (uv_map * 255).astype(np.uint8)
Image.fromarray(img, "RGB").save("voronoi_uv_puzzle.png")

print("Saved voronoi_uv_puzzle.png")
