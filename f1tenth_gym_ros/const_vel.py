import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class ConstantVelocityNode(Node):
    def __init__(self):
        super().__init__('constant_velocity_node')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        timer_period = 0.1  # seconds (10 Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.speed = 1.0  # meters per second (adjust as needed)
        self.steering_angle = 0.0
        self.get_logger().info('lets goo 1')

    def timer_callback(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering_angle
        self.publisher_.publish(msg)
        print("lets goo 2")

def main(args=None):
    rclpy.init(args=args)
    node = ConstantVelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
