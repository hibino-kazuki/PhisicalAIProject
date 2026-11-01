"""
DQN の学習 (PyTorch)

  python3 dqn_train.py                     # 学習して dqn_model.npz を作る (既定値は採用モデルと同じ設定)
  python3 dqn_train.py --steps 1000000     # 学習ステップ数を指定
  python3 dqn_train.py --device cuda       # GPU で学習する
  python3 dqn_train.py --eval              # dqn_model.npz を評価する (PyTorch 不要)
  python3 dqn_train.py --baseline          # 比較用: これまでの比例制御 (まっすぐ向かう) を評価する (PyTorch 不要)

状態・報酬・ペナルティは dqn_common.py / dqn_env.py を参照。
Double DQN + ターゲットネットワーク + 経験再生 + n ステップ報酬 + Huber 損失 + Adam。

学習したネットワークは numpy 形式 (dqn_model.npz) にも書き出すので、
ROS 2 ノード (dqn_navigator.py) は PyTorch なしでそのまま動く。
再開・解析用に PyTorch 形式 (dqn_model.pt) も保存する。
"""
import argparse
import csv
import os
import time
import numpy as np

import dqn_common as C
import dqn_env
from dqn_env import MazeEnv

HERE = os.path.dirname(os.path.abspath(__file__))


class ReplayBuffer:
    def __init__(self, size):
        self.s = np.zeros((size, C.STATE_DIM), np.float32)
        self.s2 = np.zeros((size, C.STATE_DIM), np.float32)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.d = np.zeros(size, np.float32)
        self.g = np.zeros(size, np.float32)   # s2 の価値に掛ける割引率 (n ステップ先なら gamma^n)
        self.size, self.n, self.i = size, 0, 0

    def add(self, s, a, r, s2, d, g):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i], self.g[self.i] = s, a, r, s2, d, g
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, rng, batch):
        idx = rng.integers(0, self.n, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.g[idx]


# --- 評価 (numpy の Q ネットワークで行うので、ROS 2 ノードと同じ計算になる) ---
def baseline_action(env):
    """比較用: これまでの obstacle_avoider.py と同じ考え方 (赤い円柱にまっすぐ向かう比例制御) に一番近い行動"""
    d, a = env.red_relative()
    want_w = max(-1.0, min(1.0, 2.0 * a))
    want_v = 0.0 if abs(a) > 0.2 else max(0.0, min(C.MAX_LINEAR_SPEED, 1.0 * (d - 0.3 - 0.3)))
    best, best_cost = 0, float('inf')
    for k, (acc, w) in enumerate(C.ACTIONS):
        v_next = min(C.MAX_LINEAR_SPEED, max(0.0, env.v + acc * C.DT))
        cost = abs(w - want_w) + abs(v_next - want_v)
        if cost < best_cost:
            best, best_cost = k, cost
    return best


def evaluate(policy, episodes=100, seed=12345, lidar_scale='inverse'):
    """同じ 100 通りの配置で評価する。policy(env, state) -> 行動"""
    env = MazeEnv(seed=seed, lidar_scale=lidar_scale)
    counts = {'arrived': 0, 'hit_green': 0, 'hit_red': 0, 'hit_wall': 0, 'timeout': 0}
    times = []
    for _ in range(episodes):
        s = env.reset()
        while True:
            s, _, done, truncated, outcome = env.step(policy(env, s))
            if done or truncated:
                break
        counts[outcome] += 1
        if outcome == 'arrived':
            times.append(env.t)
    counts['mean_time'] = float(np.mean(times)) if times else float('nan')
    return counts


def greedy(params):
    return lambda env, s: int(np.argmax(C.q_values(params, s)))


def format_result(r, episodes=100):
    return (f"arrived {r['arrived']}/{episodes}, hit_green {r['hit_green']}, hit_red {r['hit_red']}, "
            f"hit_wall {r['hit_wall']}, timeout {r['timeout']}, mean time to arrive {r['mean_time']:.1f} s")


# --- 学習 (PyTorch) ---
def build_qnet(nn, hidden):
    """Q ネットワーク: 状態 → 全結合 (ReLU) 2 層 → 行動ごとの Q 値"""
    return nn.Sequential(
        nn.Linear(C.STATE_DIM, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, C.N_ACTIONS),
    )


def to_numpy_params(nn, qnet):
    """PyTorch のネットワークを dqn_common.q_values() で使う形 (W0, b0, W1, ...) に変換する"""
    params = {}
    for i, layer in enumerate(m for m in qnet if isinstance(m, nn.Linear)):
        params[f'W{i}'] = layer.weight.detach().cpu().numpy().T.copy()   # (出力, 入力) → (入力, 出力)
        params[f'b{i}'] = layer.bias.detach().cpu().numpy().copy()
    return params


def train(args):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError:
        raise SystemExit('PyTorch が見つかりません。PyTorch を入れた仮想環境を有効にしてから実行してください '
                         '(例: source ~/dqn-venv/bin/activate)。--eval / --baseline は PyTorch なしで動きます。')

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = MazeEnv(seed=args.seed, lidar_scale=args.lidar_scale)

    qnet = build_qnet(nn, args.hidden).to(device)
    target = build_qnet(nn, args.hidden).to(device)
    target.load_state_dict(qnet.state_dict())
    target.requires_grad_(False)
    opt = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer)

    # ペナルティの大きさを変えて学習する場合 (dqn_env の値を上書き)
    if args.penalty is not None:
        dqn_env.PENALTY_GREEN = dqn_env.PENALTY_RED = dqn_env.PENALTY_WALL = -abs(args.penalty)

    model_path = os.path.join(HERE, args.out)
    torch_path = os.path.splitext(model_path)[0] + '.pt'
    log_path = os.path.splitext(model_path)[0] + '_train_log.csv'
    log = open(log_path, 'w', newline='')
    writer = csv.writer(log)
    writer.writerow(['step', 'epsilon', 'arrived', 'hit_green', 'hit_red', 'hit_wall', 'timeout', 'mean_time', 'elapsed_s'])
    print(f'PyTorch {torch.__version__}, device {device}, threads {torch.get_num_threads()}', flush=True)

    def act(state):
        with torch.no_grad():
            q = qnet(torch.as_tensor(state, device=device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    best_key = None
    t0 = time.time()
    s = env.reset()
    pending = []  # n ステップ分の (状態, 行動, 報酬) をためておく
    for step in range(1, args.steps + 1):
        eps = max(args.eps_end, 1.0 - (1.0 - args.eps_end) * step / (args.eps_decay * args.steps))
        if args.curriculum:
            # 出発位置を赤い円柱の近く (3 m) から始め、学習の前半で部屋全体 (制限なし) まで広げる
            frac = step / (0.5 * args.steps)
            env.max_start_gap = None if frac >= 1.0 else 3.0 + 17.0 * frac
        a = int(rng.integers(C.N_ACTIONS)) if rng.random() < eps else act(s)
        s2, r, done, truncated, _ = env.step(a)

        # n ステップ先までの報酬をまとめて 1 つの経験にする (衝突や到達の結果が早く前の行動に伝わる)
        pending.append((s, a, r))
        if done or truncated:
            while pending:  # エピソードの終わり: 残りをすべて書き出す (timeout は終了扱いにしない)
                ret = sum(args.gamma ** k * p[2] for k, p in enumerate(pending))
                buf.add(pending[0][0], pending[0][1], ret, s2, float(done), args.gamma ** len(pending))
                pending.pop(0)
        elif len(pending) == args.n_step:
            ret = sum(args.gamma ** k * p[2] for k, p in enumerate(pending))
            buf.add(pending[0][0], pending[0][1], ret, s2, 0.0, args.gamma ** len(pending))
            pending.pop(0)
        s = env.reset() if (done or truncated) else s2

        if buf.n >= args.learn_start and step % args.train_every == 0:
            bs, ba, br, bs2, bd, bg = (torch.as_tensor(x, device=device) for x in buf.sample(rng, args.batch))
            with torch.no_grad():
                # Double DQN: 次の行動はオンラインネットワークで選び、その価値はターゲットネットワークで見積もる
                a_next = qnet(bs2).argmax(dim=1, keepdim=True)
                q_next = target(bs2).gather(1, a_next).squeeze(1)
                y = br + bg * (1.0 - bd) * q_next
            q = qnet(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, y)  # Huber 損失
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
            opt.step()
            with torch.no_grad():  # ターゲットネットワークを少しずつ追従させる
                for p, tp in zip(qnet.parameters(), target.parameters()):
                    tp.lerp_(p, args.tau * args.train_every)

        if step % args.eval_every == 0:
            params = to_numpy_params(nn, qnet)
            res = evaluate(greedy(params), episodes=args.eval_episodes, lidar_scale=args.lidar_scale)
            elapsed = time.time() - t0
            writer.writerow([step, round(eps, 3), res['arrived'], res['hit_green'], res['hit_red'],
                             res['hit_wall'], res['timeout'], round(res['mean_time'], 2), round(elapsed)])
            log.flush()
            key = (res['arrived'], -res['mean_time'] if res['arrived'] else 0)
            mark = ''
            if best_key is None or key > best_key:
                best_key = key
                C.save_params(model_path, params, args.lidar_scale)
                torch.save({'state_dict': qnet.state_dict(), 'hidden': args.hidden,
                            'lidar_scale': args.lidar_scale, 'step': step}, torch_path)
                mark = f'  <- saved {args.out}'
            print(f'[{step:>8d} steps, eps {eps:.2f}, {elapsed / 60:5.1f} min] '
                  f'{format_result(res, args.eval_episodes)}{mark}', flush=True)
    log.close()
    print(f'done. best model: {args.out} (+ {os.path.basename(torch_path)}), log: {log_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=800_000)
    p.add_argument('--buffer', type=int, default=200_000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128, help='units per hidden layer')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay', type=float, default=0.4, help='epsilon reaches eps-end after this fraction of steps')
    p.add_argument('--learn-start', type=int, default=5_000)
    p.add_argument('--train-every', type=int, default=4, help='update the network once every N environment steps')
    p.add_argument('--eval-every', type=int, default=20_000)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-step', type=int, default=3, help='n-step returns')
    p.add_argument('--penalty', type=float, default=20.0, help='collision penalty (20 -> -20)')
    p.add_argument('--out', default='dqn_model.npz', help='model file name (in this folder)')
    p.add_argument('--curriculum', action='store_true', help='start near the red cylinder, widen over the first half')
    p.add_argument('--lidar-scale', choices=C.LIDAR_SCALES, default='inverse', help='how LiDAR distances are normalized')
    p.add_argument('--device', default='cpu', help="'cpu' or 'cuda'")
    p.add_argument('--threads', type=int, default=0, help='CPU threads for PyTorch (0 = PyTorch default)')
    p.add_argument('--eval', action='store_true')
    p.add_argument('--baseline', action='store_true')
    args = p.parse_args()

    if args.baseline:
        print('baseline (straight P-control):', format_result(evaluate(lambda env, s: baseline_action(env))))
    elif args.eval:
        path = os.path.join(HERE, args.out)
        params, scale = C.load_params(path), C.load_lidar_scale(path)
        print(f'DQN model {args.out} (lidar {scale}):',
              format_result(evaluate(greedy(params), episodes=args.eval_episodes, lidar_scale=scale)))
    else:
        train(args)
