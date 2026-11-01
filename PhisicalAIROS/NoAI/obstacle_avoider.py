import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data # Gazebo通信用のQoS設定
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import sys

class ObstacleAvoider(Node):
    """
    /lidar トピックを購読し、円柱を見つけて正面を向け、
    ロボット前面が円柱の 1 メートル手前に来たところで停止するノード
    停止後も円柱を観察し続け、円柱が動かされたら再び 1 メートル手前へ移動する
    円柱が見つからない間はその場で回転して探す
    停止している間は車の上の板を立て、動いている間は板を倒す
    """
    def __init__(self):
        super().__init__('obstacle_avoider')

        # Subscriber (LiDAR)
        # Gazeboからのデータを受け取るため、qos_profile_sensor_data (Best Effort) を指定
        self.subscription = self.create_subscription(
            LaserScan,
            '/lidar',        # 第5章で設定したトピック名
            self.scan_callback,
            qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning

        # Publisher (速度指令)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publisher (板の角度指令 [rad])
        self.plate_publisher = self.create_publisher(Float64, '/plate_cmd', 10)

        # 制御で使用するTwistメッセージをインスタンス化
        self.twist_msg = Twist()
        self.plate_msg = Float64()

        # 板の角度 (ラジアン): 倒した状態 / 立てた状態
        self.plate_down = 0.0
        self.plate_up = math.pi / 2

        # 停止距離 (メートル): ロボット前面から円柱表面まで
        self.stop_distance = 0.1
        # LiDAR (x=0.2) からロボット前面 (x=0.5) までの距離 (メートル)
        self.lidar_to_front = 0.3
        # LiDAR で測った距離がこの値になったら停止する
        self.target_range = self.stop_distance + self.lidar_to_front

        # 速度の上限と比例ゲイン
        self.max_linear_speed = 2.0    # 前進・後退の最大速度 (m/s)
        self.max_angular_speed = 1.6   # 回転の最大速度 (rad/s)
        self.search_angular_speed = 1.6  # 円柱を探すときの回転速度 (rad/s)
        self.linear_gain = 1.0         # 距離の誤差 [m] → 前進速度 [m/s]
        self.angular_gain = 2.0        # 方向の誤差 [rad] → 回転速度 [rad/s]

        # 停止とみなす誤差の許容値
        self.distance_tolerance = 0.02  # メートル
        self.angle_tolerance = 0.02     # ラジアン
        # 停止中、誤差がこの値を超えたら円柱が動かされたとみなして再び動き出す
        # (停止の許容値より大きくして、停止位置付近で動いたり止まったりを繰り返さないようにする)
        self.resume_distance = 0.03   # メートル
        self.resume_angle = 0.05     # ラジアン

        # 円の当てはめで赤い円柱を見分けるための設定
        # 当てはめた円の半径が cylinder_radius ± radius_tolerance なら赤い円柱とみなす
        # (壁は直線なので半径がとても大きく、緑の円柱は 0.1 m になるので対象外になる)
        self.cylinder_radius = 0.5     # 赤い円柱の半径 (メートル)
        self.radius_tolerance = 0.1    # メートル
        self.max_fit_error = 0.02      # 当てはめ誤差 (RMS) の上限 (メートル)
        self.min_cluster_points = 5    # 当てはめに使う最少の点の数
        # 緑の円柱に隠されて左右に割れた赤い円柱を、1つにまとめる距離 (円の中心どうし、メートル)
        self.merge_distance = 0.2
        # 隣り合う点の距離がこれ以上離れていたら別の物体とみなす (メートル)
        self.cluster_gap = 0.3

        # 停止位置で止まっている間は True
        self.arrived = False
        # 円柱が見つからない状態か (ログを状態が変わった時だけ出すため)
        self.searching = False

        # 周囲の距離を出力する角度の間隔 (度) と、出力の間隔 (秒)
        self.range_log_step_deg = 30
        self.range_log_period = 1.0

    @staticmethod
    def fit_circle(points):
        """
        点の列 [(x, y), ...] に円を当てはめる (最小二乗法)。
        x^2 + y^2 + D*x + E*y + F = 0 の D, E, F を求め、(中心x, 中心y, 半径, 誤差RMS) を返す。
        一部 (弧) しか見えていなくても中心と半径が求まる。当てはめられなければ None。
        """
        n = len(points)
        # 計算を安定させるため、点の重心を原点にずらしてから解く
        mx = sum(p[0] for p in points) / n
        my = sum(p[1] for p in points) / n
        sxx = sxy = syy = sx = sy = sxz = syz = sz = 0.0
        for px, py in points:
            x, y = px - mx, py - my
            z = x * x + y * y
            sxx += x * x; sxy += x * y; syy += y * y
            sx += x; sy += y
            sxz += x * z; syz += y * z; sz += z
        # 正規方程式 [[sxx sxy sx] [sxy syy sy] [sx sy n]] [D E F]^T = -[sxz syz sz]^T をクラメルの公式で解く
        a = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]]
        b = [-sxz, -syz, -sz]

        def det3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        det = det3(a)
        if abs(det) < 1e-12:
            return None  # 点が一直線上に並んでいる (壁など)
        sol = []
        for col in range(3):
            m = [row[:] for row in a]
            for row in range(3):
                m[row][col] = b[row]
            sol.append(det3(m) / det)
        d, e, f = sol
        cx, cy = -d / 2, -e / 2
        r2 = cx * cx + cy * cy - f
        if r2 <= 0:
            return None
        r = math.sqrt(r2)
        rms = math.sqrt(sum((math.hypot(px - mx - cx, py - my - cy) - r) ** 2 for px, py in points) / n)
        return cx + mx, cy + my, r, rms

    def find_cylinder(self, msg):
        """
        スキャンを「つながった点のかたまり (クラスタ)」に分け、それぞれに円を当てはめる。
        半径が赤い円柱 (0.5 m) に近いクラスタを赤い円柱とみなす。
        緑の円柱に隠されて左右に割れていても、各部分の円の中心が一致するので1つにまとめて当てはめ直す。
        見つかれば (方向[rad], 表面までの距離[m]) を、なければ None を返す。
        """
        ranges = msg.ranges
        n = len(ranges)
        valid = [math.isfinite(r) and msg.range_min <= r <= msg.range_max for r in ranges]

        # --- クラスタ分け (index の範囲 [start, end] のリスト) ---
        clusters = []
        start = None
        for i in range(n):
            if not valid[i]:
                if start is not None:
                    clusters.append([start, i - 1])
                    start = None
                continue
            if start is None:
                start = i
            elif abs(ranges[i] - ranges[i - 1]) > self.cluster_gap:
                clusters.append([start, i - 1])
                start = i
        if start is not None:
            clusters.append([start, n - 1])

        # 360度スキャンなので、最後と最初のクラスタがつながっていれば1つにまとめる
        if (len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][1] == n - 1
                and abs(ranges[0] - ranges[n - 1]) <= self.cluster_gap):
            clusters[0][0] = clusters[-1][0] - n  # 負の index は末尾からの位置
            clusters.pop()

        # --- クラスタごとに円を当てはめ、赤い円柱の候補を集める ---
        candidates = []  # [(中心x, 中心y, 点の列), ...]  座標は LiDAR から見た x (前) / y (左)
        for first, last in clusters:
            if last - first + 1 < self.min_cluster_points:
                continue
            points = []
            for i in range(first, last + 1):
                a = msg.angle_min + i * msg.angle_increment  # 負の index でも角度は正しく求まる
                r = ranges[i % n]
                points.append((r * math.cos(a), r * math.sin(a)))
            fit = self.is_red_cylinder(points)
            if fit is not None:
                candidates.append((fit[0], fit[1], points))

        # --- 中心が近い候補 (割れた同じ円柱) をまとめて当てはめ直す ---
        best = None
        used = [False] * len(candidates)
        for k, (cx, cy, points) in enumerate(candidates):
            if used[k]:
                continue
            used[k] = True
            merged = list(points)
            for j in range(k + 1, len(candidates)):
                if not used[j] and math.hypot(candidates[j][0] - cx, candidates[j][1] - cy) < self.merge_distance:
                    used[j] = True
                    merged += candidates[j][2]
            fit = self.is_red_cylinder(merged)
            if fit is None:
                continue
            cx, cy, _, _ = fit

            # 方向は円の中心の方向、距離は表面まで (中心までの距離 - 半径)
            # 一部が隠れていても、中心から求めるので偏らない
            angle = math.atan2(cy, cx)
            distance = math.hypot(cx, cy) - self.cylinder_radius

            # 複数見つかった場合は一番近いものを選ぶ
            if best is None or distance < best[1]:
                best = (angle, distance)
        return best

    def is_red_cylinder(self, points):
        """
        点の列に円を当てはめ、赤い円柱の表面らしければ当てはめ結果 (中心x, 中心y, 半径, 誤差) を、違えば None を返す。
        """
        fit = self.fit_circle(points)
        if fit is None:
            return None
        cx, cy, r, rms = fit
        if abs(r - self.cylinder_radius) > self.radius_tolerance or rms > self.max_fit_error:
            return None
        # 円の中心が点より奥にある (LiDAR から見て外側の面を見ている) ことを確認する
        # 部屋の隅のようなへこんだ形を円と間違えないため
        mean_range = sum(math.hypot(px, py) for px, py in points) / len(points)
        if math.hypot(cx, cy) <= mean_range:
            return None
        return fit

    def log_ranges(self, msg):
        """
        正面 (0度) から左回りに range_log_step_deg 度ごとの距離を出力する。
        距離は LiDAR の位置から測った値 (メートル)、inf は 10 m 以内に何もないことを表す。
        """
        n = len(msg.ranges)
        parts = []
        for deg in range(0, 360, self.range_log_step_deg):
            # -180 ~ +180 度に直して、その角度に一番近い光線の index を求める
            angle = math.radians(deg if deg <= 180 else deg - 360)
            i = round((angle - msg.angle_min) / msg.angle_increment)
            i = max(0, min(n - 1, i))
            parts.append(f'{deg if deg <= 180 else deg - 360}deg={msg.ranges[i]:.2f}')
        self.get_logger().info('Ranges: ' + ' '.join(parts),
                               throttle_duration_sec=self.range_log_period)

    def scan_callback(self, msg):
        """
        LiDARデータを受信するたびに呼び出される制御ループ
        """
        if not msg.ranges:
            self.get_logger().warn('Received empty scan data.')
            return

        # --- 周囲の距離を出力 (正面から30度ごと) ---
        self.log_ranges(msg)

        # --- 円柱の検出 ---
        # 停止中も毎回スキャンし、円柱までの距離と方向を観察し続ける
        # 第5章のSDF設定では、LiDARは360度(-pi ~ +pi)設定です。
        # 角度 0 (配列の真ん中) が「正面」、正の角度が「左」になります。
        cylinder = self.find_cylinder(msg)

        # --- 制御ロジック ---
        if cylinder is None:
            # 円柱なし: その場で左に回転して探す
            if not self.searching:
                self.get_logger().info('Cylinder not found. Searching.')
            self.searching = True
            self.arrived = False
            self.twist_msg.linear.x = 0.0
            self.twist_msg.angular.z = self.search_angular_speed
        else:
            self.searching = False
            angle, distance = cylinder
            distance_error = distance - self.target_range

            if self.arrived and (abs(distance_error) > self.resume_distance or abs(angle) > self.resume_angle):
                # 円柱が動かされた: 再び 1 メートル手前へ移動する
                self.get_logger().info(
                    f'Cylinder moved ({math.degrees(angle):.1f} deg, '
                    f'{distance - self.lidar_to_front:.2f} m from front). Moving again.')
                self.arrived = False

            if self.arrived:
                # 円柱は動いていない: 停止を続ける
                self.twist_msg.linear.x = 0.0
                self.twist_msg.angular.z = 0.0
            elif abs(distance_error) < self.distance_tolerance and abs(angle) < self.angle_tolerance:
                # 円柱に正面を向け、1 メートル手前に到達: 停止
                self.get_logger().info(
                    f'Arrived: {distance - self.lidar_to_front:.2f} m in front of the cylinder. '
                    'Stopping and raising the plate.')
                self.arrived = True
                self.twist_msg.linear.x = 0.0
                self.twist_msg.angular.z = 0.0
            else:
                # 円柱の方向へ向きを合わせる (比例制御)
                self.twist_msg.angular.z = max(-self.max_angular_speed,
                                               min(self.max_angular_speed, self.angular_gain * angle))
                if abs(angle) > 0.2:
                    # 向きが大きくずれている間は前後に動かず回転だけする
                    self.twist_msg.linear.x = 0.0
                else:
                    # 停止距離に近づく (行き過ぎたら後退する)
                    self.twist_msg.linear.x = max(-self.max_linear_speed,
                                                  min(self.max_linear_speed, self.linear_gain * distance_error))
                self.get_logger().info(
                    f'Cylinder at {math.degrees(angle):.1f} deg, '
                    f'{distance - self.lidar_to_front:.2f} m from front. Approaching.')

        # 決定した速度指令をパブリッシュ
        self.publisher_.publish(self.twist_msg)

        # 停止中は板を立て、動いている間は倒す (毎回送り、取りこぼしても次で反映されるようにする)
        self.plate_msg.data = self.plate_up if self.arrived else self.plate_down
        self.plate_publisher.publish(self.plate_msg)


def main(args=None):
    try:
        rclpy.init(args=args)

        obstacle_avoider_node = ObstacleAvoider()

        # ノードを実行し、コールバックが呼び出されるようにする
        rclpy.spin(obstacle_avoider_node)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        # ノードのクリーンアップ
        if rclpy.ok():
            obstacle_avoider_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
