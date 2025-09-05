import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class StanleyControllerNode(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        self.k = 3
        self.max_steering = np.radians(25)
        
        self.v_min = 1
        self.v_max = 10.0
        self.base_speed = 7.0
        
        self.path = self.load_path()
        
        if len(self.path[0]) == 0:
            self.get_logger().error('Failed to load path!')
            return

        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        self.get_logger().info(f'Stanley Controller initialized with {len(self.path[0])} waypoints')
        self.get_logger().info(f'Speed: base={self.base_speed}, min={self.v_min}, max={self.v_max}')

    def load_path(self):
        try:
            data = np.loadtxt('/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/spilberg_centerline.csv',
                             delimiter=',', dtype=float)
            
            x_path = data[:, 0]
            y_path = data[:, 1]
            
            # Calculate path headings from consecutive points
            dx = np.diff(x_path)
            dy = np.diff(y_path)
            path_yaw = np.arctan2(dy, dx)
            path_yaw = np.append(path_yaw, path_yaw[-1])  # Duplicate last heading
            
            self.get_logger().info(f'CSV data shape: {data.shape}')
            self.get_logger().info(f'Path length: {len(x_path)} waypoints')
            
        except Exception as e:
            self.get_logger().error(f'Error loading CSV: {str(e)}')
            return np.array([]), np.array([]), np.array([])
        
        return x_path, y_path, path_yaw

    def calculate_curvature(self, idx):
        """Calculate local curvature from path geometry"""
        x_path, y_path, path_yaw = self.path
        
        if idx == 0 or idx >= len(path_yaw) - 1:
            return 0.0
        
        # Simple curvature approximation using heading change
        heading_change = abs(self.normalize_angle(path_yaw[idx+1] - path_yaw[idx-1]))
        
        # Distance between points
        dx = x_path[idx+1] - x_path[idx-1]
        dy = y_path[idx+1] - y_path[idx-1]
        distance = np.hypot(dx, dy)
        
        if distance < 0.1:
            return 0.0
        
        curvature = heading_change / distance
        return curvature

    def speed_control(self, curvature, cross_track_error):
        """Simple speed control based on curvature and cross-track error"""
        
        # Speed reduction based on curvature
        if curvature > 0.3:
            curve_speed = self.v_min
        elif curvature > 0.1:
            curve_speed = self.v_min + (self.base_speed - self.v_min) * (0.3 - curvature) / 0.2
        else:
            curve_speed = self.base_speed
        
        # Speed reduction based on cross-track error
        abs_cte = abs(cross_track_error)
        if abs_cte > 1.0:
            cte_speed = self.v_min
        elif abs_cte > 0.5:
            cte_speed = self.base_speed * (1.5 - abs_cte) / 1.0
        else:
            cte_speed = self.v_max
        
        # Take minimum of both constraints
        target_speed = min(curve_speed, cte_speed)
        return np.clip(target_speed, self.v_min, self.v_max)

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
        x_path, y_path, path_yaw = self.path

        dx = x_path - x
        dy = y_path - y
        d = np.hypot(dx, dy)
        idx = np.argmin(d)
        
        path_heading = path_yaw[idx]
        heading_error = self.normalize_angle(path_heading - yaw)
        cross_track_error = -np.sin(yaw) * dx[idx] + np.cos(yaw) * dy[idx]

        v_safe = max(abs(v), 0.1)
        cte_term = np.arctan2(self.k * cross_track_error, v_safe)
        delta = heading_error + cte_term
        delta = np.clip(delta, -self.max_steering, self.max_steering)
        
        # Calculate local curvature and determine speed
        curvature = self.calculate_curvature(idx)
        target_speed = self.speed_control(curvature, cross_track_error)
        
        if idx % 5 == 0:
            self.get_logger().info(
                f'WP {idx}: curvature={curvature:.4f}, '
                f'target_speed={target_speed:.2f} m/s, '
                f'steering={np.degrees(delta):.1f}°, '
                f'cte={cross_track_error:.3f}m'
            )
        
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
