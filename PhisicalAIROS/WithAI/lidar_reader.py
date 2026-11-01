import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data # センサー用QoS(Best Effort)設定
from sensor_msgs.msg import LaserScan

class LidarReader(Node):
    def __init__(self):
        super().__init__('lidar_reader')
        
        # QoSプロファイルに 'sensor_data' を指定して、Gazeboからのデータを受け取れるようにする
        # これを指定しないとデータを受信できません
        self.subscription = self.create_subscription(
            LaserScan,
            '/lidar',
            self.listener_callback,
            qos_profile_sensor_data)

    def listener_callback(self, msg):
        # データの中央（正面）の距離を取得
        mid_index = len(msg.ranges) // 2
        front_dist = msg.ranges[mid_index]
        
        # ログ表示
        if front_dist == float('inf'):
             self.get_logger().info('Front: Infinity (No obstacle)')
        else:
             self.get_logger().info(f'Obstacle Distance: {front_dist:.2f} m')

def main():
    rclpy.init()
    node = LidarReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()