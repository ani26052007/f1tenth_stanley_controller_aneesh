import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class StanleyControllerNode(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        # Parameters
        self.k = 1.0              # control gain
        self.target_speed = 2.0   # m/s
        self.path = self.load_path()  # reference trajectory

        # Subscribers & publishers
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/ego_racecar/drive', 10)

    def load_path(self):
        # Load CSV data
        data = np.loadtxt('/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/csv_spielberg_map.csv',
                         delimiter=';',
                         comments='#',
                         skiprows=2,
                         dtype=float)
        
        x_path = data[:, 0]
        y_path = data[:, 1]
        
        # Calculate path headings
        path_yaw = np.zeros(len(x_path))
        for i in range(len(x_path) - 1):
            dx = x_path[i + 1] - x_path[i]
            dy = y_path[i + 1] - y_path[i]
            path_yaw[i] = np.arctan2(dy, dx)
        
        # Set last heading same as second-to-last
        path_yaw[-1] = path_yaw[-2]
        
        return x_path, y_path, path_yaw

    def odom_callback(self, msg):
        # Position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Orientation → yaw
        q = msg.pose.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Velocity
        v = msg.twist.twist.linear.x

        # Compute steering
        steer = self.stanley_control(x, y, yaw, v)

        # Publish command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.target_speed
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)

    def stanley_control(self, x, y, yaw, v):
        x_path, y_path, path_yaw = self.path

        # Find nearest path point
        dx = x_path - x  # Fixed: was y_path - x
        dy = y_path - y  # Fixed: was x_path - y
        d = np.hypot(dx, dy)
        idx = np.argmin(d)

        # Heading error
        path_heading = path_yaw[idx]
        heading_error = self.normalize_angle(path_heading - yaw)

        # Cross-track error with correct sign
        cross_track_error = d[idx]
        
        # Calculate perpendicular vector to vehicle heading
        perp_vec = [np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2)]
        
        # Determine sign of cross-track error
        error_sign = np.sign(dx[idx] * perp_vec[0] + dy[idx] * perp_vec[1])
        cross_track_error *= error_sign

        # Stanley control law
        cte_term = np.arctan2(self.k * cross_track_error, v + 1e-5)
        delta = heading_error + cte_term
        
        return self.normalize_angle(delta)

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = StanleyControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
