import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class StanleyControllerNode(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        # Parameters
        self.k = 2
        self.target_speed = 2.0
        self.max_steering = np.radians(25)
        self.min_speed = 0.1
        
        self.path = self.load_path()
        
        if len(self.path[0]) == 0:
            self.get_logger().error('Failed to load path!')
            return

        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        self.get_logger().info(f'Stanley Controller initialized with {len(self.path[0])} waypoints')

    def load_path(self):
        try:
            data = np.loadtxt('/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/csv_spielberg_map.csv',
                             delimiter=';', comments='#', skiprows=2, dtype=float)
            
            # Correct column mapping for your CSV format:
            # x_m; y_m; vx_mps; psi_rad; kappa_radpm
            x_path = data[:, 0]      # x_m (column 0)
            y_path = data[:, 1]      # y_m (column 1)  
            vx_path = data[:, 2]     # vx_mps (column 2) - FIXED!
            path_yaw = data[:, 3]    # psi_rad (column 3)
            curvature = data[:, 4]   # kappa_radpm (column 4)
            
            self.get_logger().info(f'CSV data shape: {data.shape}')
            self.get_logger().info(f'Speed range: {vx_path.min():.1f}-{vx_path.max():.1f} m/s')
            
        except Exception as e:
            self.get_logger().error(f'Error loading CSV: {str(e)}')
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        return x_path, y_path, vx_path, path_yaw, curvature

    def odom_callback(self, msg):
        if len(self.path[0]) == 0:
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        v = msg.twist.twist.linear.x

        steer, target_speed = self.stanley_control(x, y, yaw, v)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = target_speed
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)

    def stanley_control(self, x, y, yaw, v):
        x_path, y_path, vx_path, path_yaw, curvature = self.path

        dx = x_path - x
        dy = y_path - y
        d = np.hypot(dx, dy)
        idx = np.argmin(d)
        
        path_heading = path_yaw[idx]
        heading_error = self.normalize_angle(path_heading - yaw)
        cross_track_error = -np.sin(yaw) * dx[idx] + np.cos(yaw) * dy[idx]

        # Stanley control law
        v_safe = max(abs(v), self.min_speed)
        cte_term = np.arctan2(self.k * cross_track_error, v_safe)
        delta = heading_error + cte_term
        delta = np.clip(delta, -self.max_steering, self.max_steering)
        
        # Enhanced speed control for turns
        current_curvature = abs(curvature[idx])
        # reference_speed = vx_path[idx]  # Now correctly accessing column 2
        max_speed = 10
        
        # More aggressive speed reduction for turns
        if current_curvature > 0.0002:    # Sharp turn (adjusted for your curvature scale)
            speed_factor = 0.7
            self.get_logger().info(f'Turn detected: curvature={current_curvature:.6f}, reducing speed')
        elif current_curvature > 0.0001:  # Moderate turn
            speed_factor = 0.8
        elif current_curvature > 0.00005: # Gentle turn
            speed_factor = 0.9
        else:                             # Straight
            speed_factor = 1.0
        
        target_speed = max_speed * speed_factor
        target_speed = np.clip(target_speed, 2.0, 6.0)  # Reasonable speed limits
        
        return delta, target_speed

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = StanleyControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
