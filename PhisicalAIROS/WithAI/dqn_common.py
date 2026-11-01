"""
DQN で使う共通部分 (学習用シミュレータ dqn_env.py / 学習 dqn_train.py / ROS 2 ノード dqn_navigator.py)

学習時と実行時で「状態の作り方」「行動の意味」「ネットワークの計算」が食い違わないよう、ここにまとめる。
numpy だけで動く (PyTorch 不要)。
"""
import math
import numpy as np

# --- 制御周期と車の動き ---
DT = 0.1                 # 制御周期 (秒)。LiDAR の更新周期 (10 Hz) に合わせる
MAX_LINEAR_SPEED = 2.0   # 前進の最大速度 (m/s)
MAX_ANGULAR_SPEED = 1.0  # 回転の最大速度 (rad/s)
LINEAR_ACCEL = 1.0       # 1 秒あたりの速度変化 (m/s^2)。急には止まれない

# --- 行動: (加速度 [m/s^2], 回転速度 [rad/s]) の組み合わせ 3 x 5 = 15 通り ---
ACCELS = (-LINEAR_ACCEL, 0.0, LINEAR_ACCEL)                   # 減速 / そのまま / 加速
TURNS = (-1.0, -0.5, 0.0, 0.5, 1.0)                          # 右 ... 左 (rad/s)
ACTIONS = [(a, w) for a in ACCELS for w in TURNS]
N_ACTIONS = len(ACTIONS)

# --- 状態 ---
LIDAR_BINS = 72          # 全方位 360 度を 5 度ずつに分け、各区間の最短距離を使う
LIDAR_MAX_RANGE = 10.0   # LiDAR の最大距離 (m)。inf はこの値として扱う
# LiDAR の距離を 0 ~ 1 に直す方法
#   'linear' : 距離 / 10 m            (遠くも近くも同じ細かさ)
#   'inverse': 0.5 m / 距離 (上限 1)  (近いほど細かい: 0.5 m→1.0, 1 m→0.5, 5 m→0.1)。ぶつかりそうな近さを見分けやすい
LIDAR_SCALES = ('linear', 'inverse')
INVERSE_REF = 0.5
STATE_DIM = LIDAR_BINS + 3 + 2   # LiDAR + 赤い円柱 (距離, sin角度, cos角度) + 車の速度 (前進, 回転)


def apply_action(v, action):
    """行動を適用した後の (前進速度, 回転速度) を返す。前進速度は加速度の制限つきで変化する"""
    accel, w = ACTIONS[action]
    v = min(MAX_LINEAR_SPEED, max(0.0, v + accel * DT))
    return v, w


def lidar_bins(ranges, angle_min, angle_increment, scale='linear'):
    """
    LiDAR の全方位スキャンを LIDAR_BINS 個の区間に分け、区間ごとの最短距離を返す (0 ~ 1 に正規化、scale で方法を選ぶ)。
    区間 k は角度 [-180 + 5k, -180 + 5(k+1)) 度。最短距離を使うので細い円柱も見落とさない。
    """
    r = np.asarray(ranges, dtype=np.float64)
    # +inf / NaN は「何もない」、-inf は「最小距離より近い」を表すので 0 とする
    r = np.where(np.isneginf(r), 0.0, np.where(np.isfinite(r), r, LIDAR_MAX_RANGE))
    r = np.clip(r, 0.0, LIDAR_MAX_RANGE)
    angles = angle_min + np.arange(len(r)) * angle_increment
    angles = (angles + math.pi) % (2 * math.pi)                     # 0 ~ 2pi (0 が真後ろ)
    k = np.minimum((angles / (2 * math.pi) * LIDAR_BINS).astype(int), LIDAR_BINS - 1)
    bins = np.full(LIDAR_BINS, LIDAR_MAX_RANGE)
    np.minimum.at(bins, k, r)
    if scale == 'inverse':
        return np.minimum(1.0, INVERSE_REF / np.maximum(bins, 1e-3))
    return bins / LIDAR_MAX_RANGE


def make_state(bins, red_distance, red_angle, v, w):
    """
    状態ベクトルを作る。
    bins: lidar_bins() の結果 / red_distance: LiDAR から赤い円柱の表面までの距離 (m)
    red_angle: LiDAR から見た赤い円柱の中心の方向 (rad, 左が正) / v, w: 車の前進速度と回転速度
    """
    red = [min(red_distance, LIDAR_MAX_RANGE) / LIDAR_MAX_RANGE, math.sin(red_angle), math.cos(red_angle)]
    speed = [v / MAX_LINEAR_SPEED, w / MAX_ANGULAR_SPEED]
    return np.concatenate([bins, red, speed]).astype(np.float32)


# --- Q ネットワーク (全結合 3 層、ReLU) ---
def init_params(rng, sizes=(STATE_DIM, 128, 128, N_ACTIONS)):
    params = {}
    for i in range(len(sizes) - 1):
        params[f'W{i}'] = (rng.standard_normal((sizes[i], sizes[i + 1])) * math.sqrt(2.0 / sizes[i])).astype(np.float32)
        params[f'b{i}'] = np.zeros(sizes[i + 1], dtype=np.float32)
    return params


def q_values(params, x):
    """Q 値を計算する。x: (状態数, STATE_DIM) または (STATE_DIM,)"""
    h = np.asarray(x, dtype=np.float32)
    n_layers = sum(1 for k in params if k.startswith('W'))
    for i in range(n_layers):
        h = h @ params[f'W{i}'] + params[f'b{i}']
        if i < n_layers - 1:
            h = np.maximum(h, 0.0)
    return h


def save_params(path, params, lidar_scale='linear'):
    """重みと一緒に、学習時の LiDAR の正規化方法も保存する (実行時に同じ方法で状態を作るため)"""
    np.savez(path, lidar_scale=np.array(lidar_scale), **params)


def load_params(path):
    """重み (W*, b*) だけを読み込む"""
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k[0] in 'Wb' and k[1:].isdigit()}


def load_lidar_scale(path):
    """学習時の LiDAR の正規化方法 (保存されていない古いモデルは 'linear')"""
    with np.load(path) as data:
        return str(data['lidar_scale']) if 'lidar_scale' in data.files else 'linear'
