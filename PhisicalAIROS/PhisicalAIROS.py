
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys

class CirclePublisher(Node):
    """
    一定の速度指令（Twist）をパブリッシュしてロボットを円運動させるノード
    """
    def __init__(self):
        super().__init__('circle_publisher')
        
        # /cmd_vel トピックに対して Twist 型のメッセージをパブリッシュする Publisher を作成
        self.publisher_ = self.create_publisher(Twist, '/model/vehicle_blue/cmd_vel', 10)
        
        # 0.5秒ごとに timer_callback メソッドを呼び出すタイマーを作成
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # パブリッシュするTwistメッセージの定義
        self.twist_msg = Twist()
        self.twist_msg.linear.x = 0.2   # 前進速度 (m/s)
        self.twist_msg.linear.y = 0.0
        self.twist_msg.linear.z = 0.0
        self.twist_msg.angular.x = 0.0
        self.twist_msg.angular.y = 0.0
        self.twist_msg.angular.z = 0.3  # 旋回速度 (rad/s)

    def timer_callback(self):
        """
        タイマーによって定期的に呼び出され、Twistメッセージをパブリッシュする
        """
        # メッセージをパブリッシュ
        self.publisher_.publish(self.twist_msg)
        
        # ターミナルへのログ出力
        self.get_logger().info(f'Publishing: linear.x={self.twist_msg.linear.x}, angular.z={self.twist_msg.angular.z}')

def main(args=None):
    try:
        rclpy.init(args=args)
        
        circle_publisher_node = CirclePublisher()
        
        # ノードを実行し、コールバックが呼び出されるようにする
        # Ctrl+Cが押されるまでスピン（待機）
        rclpy.spin(circle_publisher_node)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        # ノードのクリーンアップ
        if rclpy.ok():
            circle_publisher_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()