import math
import os
import sys
import numpy as np
import rclpy

import dqn_common as C
from dqn_env import footprint_gap, LIDAR_OFFSET, RED_RADIUS, ARRIVE_GAP, ARRIVE_SPEED
from obstacle_avoider import ObstacleAvoider


class DqnNavigator(ObstacleAvoider):
    """
    学習済みの DQN モデル (dqn_model.npz) で、緑の円柱や壁を避けながら赤い円柱まで移動するノード
    ・赤い円柱の検出 (円の当てはめ)、/cmd_vel と /plate_cmd の送信は obstacle_avoider.py と同じ
    ・赤い円柱の近く (車体との隙間 0.5 m 以内) で速度 0.3 m/s 以下になったら到達とみなして停止し、板を立てる
    ・停止後に赤い円柱が動かされたら、再び DQN で移動する
    ・緑の円柱に隠れて赤い円柱が見えない間は、最後に見えた位置を自分の動きに合わせて更新して使う
    """
    def __init__(self):
        super().__init__('dqn_navigator')

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dqn_model.npz')
        self.params = C.load_params(model_path)
        self.lidar_scale = C.load_lidar_scale(model_path)  # 学習時と同じ方法で LiDAR の距離を正規化する
        self.get_logger().info(f'Loaded DQN model: {model_path} (lidar scale: {self.lidar_scale})')

        # 車が出している速度 (学習時と同じく、加速度の制限つきで変化させる)
        self.v = 0.0
        self.w = 0.0
        # 赤い円柱の中心 (LiDAR から見た x: 前 / y: 左, メートル)。まだ見つけていなければ None
        self.red_center = None
        # 停止したときの赤い円柱の中心 (動かされたかの判定に使う)
        self.arrived_center = None
        # 停止中、赤い円柱の中心がこれ以上動いたら再び動き出す (メートル)
        self.resume_move = 0.1

    def update_red_center(self, msg):
        """赤い円柱の位置を更新する。見えなければ、自分の動きの分だけ前回の位置をずらす"""
        cylinder = self.find_cylinder(msg)
        if cylinder is not None:
            angle, distance = cylinder
            r = distance + self.cylinder_radius
            self.red_center = (r * math.cos(angle), r * math.sin(angle))
            return True
        if self.red_center is not None:
            # 車の中心から見た位置に直し、前回の指令で車が動いた分を差し引いてから LiDAR から見た位置に戻す
            x, y = self.red_center[0] + LIDAR_OFFSET, self.red_center[1]
            x -= self.v * C.DT
            d = -self.w * C.DT
            x, y = x * math.cos(d) - y * math.sin(d), x * math.sin(d) + y * math.cos(d)
            self.red_center = (x - LIDAR_OFFSET, y)
        return False

    def scan_callback(self, msg):
        if not msg.ranges:
            self.get_logger().warn('Received empty scan data.')
            return
        self.log_ranges(msg)
        visible = self.update_red_center(msg)

        if self.red_center is None:
            # まだ一度も見つけていない: 減速しながらその場で回転して探す
            if not self.searching:
                self.get_logger().info('Red cylinder not found yet. Searching.')
            self.searching = True
            self.v = max(0.0, self.v - C.LINEAR_ACCEL * C.DT)
            self.w = 0.5
        else:
            self.searching = False
            cx, cy = self.red_center
            # 車体と赤い円柱の表面の隙間 (学習時の到達判定と同じ計算)
            gap = float(footprint_gap(0.0, 0.0, 0.0, cx + LIDAR_OFFSET, cy, RED_RADIUS))

            if self.arrived and visible and math.hypot(cx - self.arrived_center[0],
                                                       cy - self.arrived_center[1]) > self.resume_move:
                self.get_logger().info('Red cylinder moved. Moving again with DQN.')
                self.arrived = False

            if self.arrived:
                self.v, self.w = 0.0, 0.0
            elif gap <= ARRIVE_GAP and self.v <= ARRIVE_SPEED:
                self.get_logger().info(f'Arrived: gap {gap:.2f} m to the red cylinder. '
                                       'Stopping and raising the plate.')
                self.arrived = True
                self.arrived_center = self.red_center
                self.v, self.w = 0.0, 0.0
            else:
                # --- DQN で行動を選ぶ ---
                bins = C.lidar_bins(msg.ranges, msg.angle_min, msg.angle_increment, self.lidar_scale)
                distance = math.hypot(cx, cy) - RED_RADIUS
                angle = math.atan2(cy, cx)
                state = C.make_state(bins, distance, angle, self.v, self.w)
                action = int(np.argmax(C.q_values(self.params, state)))
                self.v, self.w = C.apply_action(self.v, action)
                self.get_logger().info(
                    f'red {math.degrees(angle):+.0f} deg {distance:.2f} m'
                    f'{"" if visible else " (hidden, estimated)"} -> v {self.v:.1f} m/s, w {self.w:+.1f} rad/s')

        self.twist_msg.linear.x = self.v
        self.twist_msg.angular.z = self.w
        self.publisher_.publish(self.twist_msg)
        self.plate_msg.data = self.plate_up if self.arrived else self.plate_down
        self.plate_publisher.publish(self.plate_msg)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = DqnNavigator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
