"""
DQN の学習用 2 次元シミュレータ

maze_bot.sdf の世界 (16 m 四方の壁、緑の円柱 20 本、赤い円柱) と車の大きさ・LiDAR を再現する。
Gazebo より何百倍も速く動くので、ここで学習してから Gazebo / ROS 2 で使う。

・車: 1.0 x 0.96 m の長方形 (車輪を含む)、LiDAR は中心から 0.2 m 前、全方位 1080 本
・赤い円柱: 半径 0.5 m、エピソードごとにランダムな位置 (手で動かされる想定)
・緑の円柱: maze_bot.sdf に書かれた位置と半径
・車の出発位置と向きもランダム、静止状態から出発する
"""
import math
import os
import re
import numpy as np

import dqn_common as C

WALL_INNER = 7.9          # 壁の内側の面 (壁の中心 ±8 m、厚さ 0.2 m)
RED_RADIUS = 0.5
ROBOT_HALF_LENGTH = 0.5   # 車の前後方向の半分 (キャスターを含む)
ROBOT_HALF_WIDTH = 0.48   # 車の左右方向の半分 (車輪を含む)
LIDAR_OFFSET = 0.2        # 車の中心から LiDAR までの距離 (前方向)
LIDAR_SAMPLES = 1080
LIDAR_ANGLE_MIN = -3.14159
LIDAR_ANGLE_MAX = 3.14159
LIDAR_ANGLE_INC = (LIDAR_ANGLE_MAX - LIDAR_ANGLE_MIN) / (LIDAR_SAMPLES - 1)
LIDAR_RANGE_MIN = 0.15

# --- 到達・衝突の判定 ---
ARRIVE_GAP = 0.5          # 車体と赤い円柱の表面の隙間がこれ以下で、
ARRIVE_SPEED = 0.3        # 速度がこれ以下なら「到達」 (m/s)
MAX_EPISODE_TIME = 60.0   # これを超えたら打ち切り (秒)

# --- 報酬 ---
REWARD_TIME = -0.02       # 1 ステップ (0.1 秒) ごと: 到達までの時間が短いほど合計の報酬が大きくなる
REWARD_PROGRESS = 1.0     # 赤い円柱に 1 m 近づくごとに (学習を進みやすくするための補助)
REWARD_ARRIVE = 10.0      # 赤い円柱に到達
PENALTY_GREEN = -10.0     # 緑の円柱に衝突
PENALTY_RED = -10.0       # 赤い円柱に衝突 (速度の出しすぎで止まりきれない場合など)
PENALTY_WALL = -10.0      # 壁に衝突


def load_green_cylinders(sdf_path):
    """maze_bot.sdf から緑の円柱 (cylinder_01 ~ 20) の位置と半径を読み取る"""
    src = open(sdf_path, encoding='utf-8').read()
    greens = []
    for body in re.findall(r'<model name="cylinder_\d+">(.*?)</model>', src, re.S):
        x, y = re.search(r'<pose>(\S+) (\S+) ', body).groups()
        r = re.search(r'<radius>(\S+)</radius>', body).group(1)
        greens.append((float(x), float(y), float(r)))
    return np.array(greens)


def footprint_gap(x, y, th, cx, cy, r):
    """車体 (長方形) と円の表面の隙間 (m)。0 以下なら接触している。cx, cy, r は配列でもよい"""
    c, s = math.cos(th), math.sin(th)
    lx = (cx - x) * c + (cy - y) * s
    ly = -(cx - x) * s + (cy - y) * c
    dx = np.maximum(np.abs(lx) - ROBOT_HALF_LENGTH, 0.0)
    dy = np.maximum(np.abs(ly) - ROBOT_HALF_WIDTH, 0.0)
    return np.hypot(dx, dy) - r


def footprint_hits_wall(x, y, th, margin=0.0):
    c, s = math.cos(th), math.sin(th)
    for sx in (-1, 1):
        for sy in (-1, 1):
            px = x + sx * ROBOT_HALF_LENGTH * c - sy * ROBOT_HALF_WIDTH * s
            py = y + sx * ROBOT_HALF_LENGTH * s + sy * ROBOT_HALF_WIDTH * c
            if abs(px) > WALL_INNER - margin or abs(py) > WALL_INNER - margin:
                return True
    return False


class MazeEnv:
    def __init__(self, sdf_path=None, seed=0, lidar_scale='linear'):
        self.lidar_scale = lidar_scale
        if sdf_path is None:
            sdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze_bot.sdf')
        self.greens = load_green_cylinders(sdf_path)
        self.rng = np.random.default_rng(seed)
        self.ray_offsets = LIDAR_ANGLE_MIN + np.arange(LIDAR_SAMPLES) * LIDAR_ANGLE_INC
        # 出発時の車と赤い円柱の隙間の上限 (m)。None なら制限なし。学習の初めは近くから始める (カリキュラム学習)
        self.max_start_gap = None

    # --- LiDAR ---
    def scan(self):
        """LiDAR の全方位スキャン (1080 本) を計算する。何もなければ inf"""
        lx = self.x + LIDAR_OFFSET * math.cos(self.th)
        ly = self.y + LIDAR_OFFSET * math.sin(self.th)
        a = self.th + self.ray_offsets
        dx, dy = np.cos(a), np.sin(a)
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.where(dx > 0, (WALL_INNER - lx) / dx, np.where(dx < 0, (-WALL_INNER - lx) / dx, np.inf))
            ty = np.where(dy > 0, (WALL_INNER - ly) / dy, np.where(dy < 0, (-WALL_INNER - ly) / dy, np.inf))
        t = np.minimum(tx, ty)
        circles = np.vstack([self.greens, [self.red_x, self.red_y, RED_RADIUS]])
        fx = lx - circles[:, 0]
        fy = ly - circles[:, 1]
        b = np.outer(dx, fx) + np.outer(dy, fy)                    # (光線, 円)
        cc = fx * fx + fy * fy - circles[:, 2] ** 2
        disc = b * b - cc
        hit = -b - np.sqrt(np.maximum(disc, 0.0))
        hit = np.where((disc >= 0) & (hit > 0), hit, np.inf)
        t = np.minimum(t, hit.min(axis=1))
        return np.where(t <= C.LIDAR_MAX_RANGE, np.maximum(t, LIDAR_RANGE_MIN), np.inf)

    def red_relative(self):
        """LiDAR から見た赤い円柱: (表面までの距離, 中心の方向)。ROS 2 ノードでは円の当てはめで同じ値を求める"""
        lx = self.x + LIDAR_OFFSET * math.cos(self.th)
        ly = self.y + LIDAR_OFFSET * math.sin(self.th)
        d = math.hypot(self.red_x - lx, self.red_y - ly) - RED_RADIUS
        a = math.atan2(self.red_y - ly, self.red_x - lx) - self.th
        return d, math.atan2(math.sin(a), math.cos(a))

    def state(self):
        bins = C.lidar_bins(self.scan(), LIDAR_ANGLE_MIN, LIDAR_ANGLE_INC, self.lidar_scale)
        d, a = self.red_relative()
        return C.make_state(bins, d, a, self.v, self.w)

    # --- エピソード ---
    def reset(self, red=None, start=None):
        """赤い円柱と車をランダムに置き直す (red=(x, y), start=(x, y, th) で指定も可)"""
        g = self.greens
        lim = WALL_INNER - RED_RADIUS - 0.3
        while True:
            self.red_x, self.red_y = red if red is not None else self.rng.uniform(-lim, lim, 2)
            # 緑の円柱から表面どうし 0.3 m 以上離す
            if red is not None or np.all(np.hypot(g[:, 0] - self.red_x, g[:, 1] - self.red_y) - g[:, 2] - RED_RADIUS > 0.3):
                break
        lim = WALL_INNER - 0.7
        while True:
            if start is not None:
                self.x, self.y, self.th = start
            elif self.max_start_gap is not None:
                # 赤い円柱のまわり (隙間 2 m ~ max_start_gap) に置く
                d = self.rng.uniform(2.0, max(2.0, self.max_start_gap)) + RED_RADIUS + ROBOT_HALF_LENGTH
                b = self.rng.uniform(-math.pi, math.pi)
                self.x, self.y = self.red_x + d * math.cos(b), self.red_y + d * math.sin(b)
                self.th = self.rng.uniform(-math.pi, math.pi)
            else:
                self.x, self.y = self.rng.uniform(-lim, lim, 2)
                self.th = self.rng.uniform(-math.pi, math.pi)
            start_gap = footprint_gap(self.x, self.y, self.th, self.red_x, self.red_y, RED_RADIUS)
            ok = (not footprint_hits_wall(self.x, self.y, self.th, margin=0.1)
                  and np.all(footprint_gap(self.x, self.y, self.th, g[:, 0], g[:, 1], g[:, 2]) > 0.3)
                  and start_gap > 2.0
                  and (self.max_start_gap is None or start_gap <= self.max_start_gap + 1.0))
            if ok or start is not None:
                break
        self.v, self.w = 0.0, 0.0     # 静止状態から出発
        self.t = 0.0
        self.prev_gap = float(footprint_gap(self.x, self.y, self.th, self.red_x, self.red_y, RED_RADIUS))
        return self.state()

    def step(self, action):
        """
        行動を 1 ステップ (0.1 秒) 実行する。
        戻り値: (次の状態, 報酬, 終了したか, 打ち切りか, 結果)
        結果は 'arrived' / 'hit_green' / 'hit_red' / 'hit_wall' / 'timeout' / None
        """
        self.v, self.w = C.apply_action(self.v, action)
        # 回転の途中の向きで進める (中点法)
        mid = self.th + self.w * C.DT / 2
        self.x += self.v * math.cos(mid) * C.DT
        self.y += self.v * math.sin(mid) * C.DT
        self.th = math.atan2(math.sin(self.th + self.w * C.DT), math.cos(self.th + self.w * C.DT))
        self.t += C.DT

        g = self.greens
        gap = float(footprint_gap(self.x, self.y, self.th, self.red_x, self.red_y, RED_RADIUS))
        reward = REWARD_TIME + REWARD_PROGRESS * (self.prev_gap - gap)
        self.prev_gap = gap

        outcome = None
        if np.any(footprint_gap(self.x, self.y, self.th, g[:, 0], g[:, 1], g[:, 2]) <= 0):
            reward += PENALTY_GREEN; outcome = 'hit_green'
        elif gap <= 0:
            reward += PENALTY_RED; outcome = 'hit_red'
        elif footprint_hits_wall(self.x, self.y, self.th):
            reward += PENALTY_WALL; outcome = 'hit_wall'
        elif gap <= ARRIVE_GAP and self.v <= ARRIVE_SPEED:
            reward += REWARD_ARRIVE; outcome = 'arrived'
        done = outcome is not None
        truncated = not done and self.t >= MAX_EPISODE_TIME - 1e-9
        if truncated:
            outcome = 'timeout'
        return self.state(), reward, done, truncated, outcome
