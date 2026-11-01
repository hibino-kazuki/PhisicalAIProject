# maze_bot：赤い円柱まで移動するロボット（Gazebo + ROS 2）

Gazebo の空間（16 m 四方、緑の円柱 20 本、赤い円柱 1 本）で、LiDAR を積んだ車が赤い円柱まで移動します。
赤い円柱の手前で止まると、車の上の板（日の丸）を立てます。赤い円柱をマウスで動かすと、車はまた移動します。

動かし方は 2 通りあります。

| 方式 | プログラム | 内容 |
|---|---|---|
| AI なし | `NoAI/obstacle_avoider.py` | 赤い円柱へまっすぐ向かう（比例制御）。緑の円柱は避けない |
| AI あり | `WithAI/dqn_navigator.py` | 強化学習（DQN）で学習したモデルで、緑の円柱や壁を避けながら向かう |

## フォルダ構成

```
.phisicalairos/
├── ReadMe.md              この説明
├── sdf/
│   ├── robot.sdf          Gazebo の世界（部屋・円柱・車・LiDAR・カメラ・板）
│   └── nippon.png         板に貼る画像（robot.sdf と同じフォルダに置く）
├── NoAI/
│   ├── obstacle_avoider.py  AI なしのノード
│   └── lidar_reader.py      LiDAR のデータを表示するだけのノード
└── WithAI/
    ├── dqn_navigator.py     AI ありのノード
    ├── dqn_model.npz        学習済みモデル（dqn_navigator.py が読み込む）
    ├── dqn_common.py        状態・行動・ネットワーク計算の共通部分
    ├── dqn_env.py           学習用の 2 次元シミュレータ
    ├── dqn_train.py         学習プログラム（PyTorch）
    └── obstacle_avoider.py  赤い円柱の検出に使う（dqn_navigator.py から読み込む）
```

## 必要なもの

- Ubuntu 24.04（WSL2 可）
- ROS 2 Jazzy
- Gazebo Harmonic（`gz sim`）
- ros_gz（`ros_gz_bridge`。ROS 2 と Gazebo の間でトピックをやり取りする）

ROS 2 のコマンドを使うターミナルでは、最初に毎回次を実行してください。

```bash
source /opt/ros/jazzy/setup.bash
```

なお、環境構築は下記を参考に実施。
https://zenn.dev/sakai13/books/0fa1d6a8c61f9b/viewer/506f7e


## 手順

ターミナルを 3 つ使います。

| ターミナル | 実行するもの |
|---|---|
| 1 | Gazebo（`gz sim -r robot.sdf`） |
| 2 | ブリッジ（ROS 2 と Gazebo をつなぐ） |
| 3 | ノード（`obstacle_avoider.py` または `dqn_navigator.py` のどちらか 1 つ） |

### 1. Gazebo を起動する（ターミナル 1）

```bash
cd ~/.phisicalairos/sdf
```

```bash
gz sim -r robot.sdf
```

- `-r` を付けると、起動と同時にシミュレーションが始まります。付けない場合は、Gazebo の画面左下の再生ボタンを押してください。
- 板に貼る画像 `nippon.png` は、`robot.sdf` と同じフォルダから読み込まれます。
- 赤い円柱を動かすには、Gazebo の画面上部の移動ツール（矢印のアイコン、またはキーボードの `T`）を選び、赤い円柱をドラッグします。

### 2. ブリッジを起動する（ターミナル 2）

ROS 2 のノードと Gazebo の間で、次の 3 つのトピックをやり取りします。

| トピック | 向き | 内容 |
|---|---|---|
| `/lidar` | Gazebo → ROS 2 | LiDAR の全方位スキャン |
| `/cmd_vel` | ROS 2 → Gazebo | 車の速度指令 |
| `/plate_cmd` | ROS 2 → Gazebo | 板の角度指令（0：倒す、1.5708：立てる） |

```bash
source /opt/ros/jazzy/setup.bash
```

```bash
ros2 run ros_gz_bridge parameter_bridge /lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan /cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist /plate_cmd@std_msgs/msg/Float64]gz.msgs.Double
```

`[` は Gazebo から ROS 2 へ、`]` は ROS 2 から Gazebo への向きを表します。

### 3-A. AI なしで動かす：obstacle_avoider.py（ターミナル 3）

```bash
source /opt/ros/jazzy/setup.bash
```

```bash
cd ~/.phisicalairos/NoAI && python3 obstacle_avoider.py
```

動き：

1. LiDAR のスキャンに円を当てはめて、半径 0.5 m の赤い円柱を探します。緑の円柱（半径 0.1 m）や壁とは半径で見分けます。緑の円柱に一部が隠れていても検出できます。
2. 見つからない間は、その場で回転して探します。
3. 赤い円柱に正面を向けて近づき、車の前面と円柱の間が `stop_distance`（現在 0.1 m）になったら止まって、板を立てます。
4. 赤い円柱が動かされると板を倒し、また近づきます。
5. 1 秒ごとに、正面から 30° ごとの距離をログに出します。

注意：赤い円柱へまっすぐ向かうので、途中に緑の円柱があるとぶつかることがあります。

主な設定（`obstacle_avoider.py` の `__init__` で変更できます）：

| 設定 | 現在の値 | 内容 |
|---|---|---|
| `stop_distance` | 0.1 m | 止まる位置（車の前面から赤い円柱の表面まで） |
| `max_linear_speed` | 2.0 m/s | 前進・後退の最大速度 |
| `max_angular_speed` | 1.6 rad/s | 回転の最大速度 |
| `search_angular_speed` | 1.6 rad/s | 赤い円柱を探すときの回転速度 |

### 3-B. AI あり（衝突回避）で動かす：dqn_navigator.py（ターミナル 3）

```bash
source /opt/ros/jazzy/setup.bash
```

```bash
cd ~/.phisicalairos/WithAI && python3 dqn_navigator.py
```

起動すると `Loaded DQN model: .../dqn_model.npz (lidar scale: inverse)` と表示されます。

動き：

1. 赤い円柱の検出は `obstacle_avoider.py` と同じ方法（円の当てはめ）です。
2. LiDAR の全方位スキャン・赤い円柱までの距離と角度・車の速度から、学習済みモデルが行動（加速・減速と回転）を選びます。緑の円柱や壁を避けながら赤い円柱へ向かいます。
3. 車体と赤い円柱の隙間が 0.5 m 以内で、速度が 0.3 m/s 以下になったら到達とみなして止まり、板を立てます。
4. 赤い円柱が動かされると（中心が 0.1 m 以上ずれると）、また移動します。
5. 緑の円柱に隠れて赤い円柱が見えない間は、最後に見えた位置を自分の動きに合わせて推定して進みます。


### 4. 止める

- ノードは `Ctrl+C` で止めます。
- Gazebo の車は、最後に受け取った速度指令で動き続けます。ノードを止めた後に車を止めるには、次を実行します。

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

- `obstacle_avoider.py` と `dqn_navigator.py` は、同時に起動しないでください（両方が `/cmd_vel` に指令を送るため）。

## 注意点

- LiDAR が測れるのは 10 m までです。赤い円柱が車から 10 m より遠いと見つけられず、その場で回転して探し続けます。その場合は赤い円柱を近くに動かしてください。
- 車が動かないとき：ブリッジが起動しているか確認してください。LiDAR のデータが届いているかは次で確認できます。

```bash
ros2 topic echo /lidar --once
```

## 補足：モデルを学習し直す（任意）

`dqn_train.py` は PyTorch を使います。PyTorch を入れた仮想環境（例：`~/dqn-venv`）を有効にしてから実行してください。

```bash
source ~/dqn-venv/bin/activate
```

```bash
cd ~/.phisicalairos/WithAI && python3 dqn_train.py --out dqn_model_new.npz
```

- 学習用シミュレータは、`sdf/robot.sdf` から緑の円柱の位置を読み込みます。
- できたモデルを使うには、`dqn_model_new.npz` を `dqn_model.npz` という名前でコピーします。
- 評価だけなら PyTorch なしで実行できます：`python3 dqn_train.py --eval`（`dqn_model.npz` を評価）
