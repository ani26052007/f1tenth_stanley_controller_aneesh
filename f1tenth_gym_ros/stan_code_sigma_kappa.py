import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class StanleyControllerNode(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        self.k = 2
        self.max_steering = np.radians(25)
        
        self.v_min = 1
        self.v_max = 8
        self.beta = 20
        self.kappa_0 = 0.25
        
        self.path = self.load_path()
        
        if len(self.path[0]) == 0:
            self.get_logger().error('Failed to load path!')
            return

        self.auto_calculate_kappa_0()

        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        self.get_logger().info(f'Stanley Controller initialized with {len(self.path[0])} waypoints')
        self.get_logger().info(f'Parameters: v_min={self.v_min}, v_max={self.v_max}, κ_0={self.kappa_0:.6f}, β={self.beta}')

    def load_path(self):
        try:
            data = np.loadtxt('/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/csv_spielberg_map.csv',
                             delimiter=';', comments='#', skiprows=2, dtype=float)
            
            x_path = data[:, 0]
            y_path = data[:, 1]
            vx_path = data[:, 2]
            path_yaw = data[:, 3]
            curvature = data[:, 4]
            
            self.get_logger().info(f'CSV data shape: {data.shape}')
            self.get_logger().info(f'Original speed range: {vx_path.min():.1f}-{vx_path.max():.1f} m/s')
            
        except Exception as e:
            self.get_logger().error(f'Error loading CSV: {str(e)}')
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        return x_path, y_path, vx_path, path_yaw, curvature

    def auto_calculate_kappa_0(self):
        if len(self.path[0]) == 0:
            return
            
        curvature = self.path[4]
        abs_curvature = np.abs(curvature)
        non_zero_curvature = abs_curvature[abs_curvature > 1e-6]
        
        if len(non_zero_curvature) == 0:
            self.get_logger().warn('No significant curvature found, using default κ_0')
            return
        
        curvature_stats = {
            'min': abs_curvature.min(),
            'max': abs_curvature.max(),
            'mean': abs_curvature.mean(),
            'median': np.median(abs_curvature),
            'percentile_25': np.percentile(abs_curvature, 25),
            'percentile_50': np.percentile(abs_curvature, 50),
            'percentile_75': np.percentile(abs_curvature, 75),
            'percentile_90': np.percentile(abs_curvature, 90)
        }
        
        self.get_logger().info('=== Curvature Analysis ===')
        for key, value in curvature_stats.items():
            self.get_logger().info(f'  {key}: {value:.6f} m⁻¹')
        
        self.kappa_0 = curvature_stats['percentile_50']
        self.kappa_0 = np.clip(self.kappa_0, 0.01, 2.0)
        
        self.get_logger().info(f'Auto-calculated κ_0 = {self.kappa_0:.6f} m⁻¹')
        self.log_speed_examples()

    def log_speed_examples(self):
        test_curvatures = [0.0, self.kappa_0/4, self.kappa_0/2, self.kappa_0, 2*self.kappa_0, 4*self.kappa_0]
        
        self.get_logger().info('=== Speed Examples ===')
        for kappa in test_curvatures:
            speed = self.curvature_based_speed(kappa)
            self.get_logger().info(f'  κ={kappa:.4f} m⁻¹ → v={speed:.2f} m/s')

    def curvature_based_speed(self, kappa):
        abs_kappa = abs(kappa)
        exponent = self.beta * (abs_kappa - self.kappa_0)
        exponent = np.clip(exponent, -500, 500)
        
        sigmoid_denominator = 1 + np.exp(exponent)
        target_speed = self.v_min + (self.v_max - self.v_min) / sigmoid_denominator
        
        return target_speed

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

        v_safe = max(abs(v), 0.1)
        cte_term = np.arctan2(self.k * cross_track_error, v_safe)
        delta = heading_error + cte_term
        delta = np.clip(delta, -self.max_steering, self.max_steering)
        
        current_curvature = curvature[idx]
        target_speed = self.curvature_based_speed(current_curvature)
        target_speed = np.clip(target_speed, 1, 8)
        
        if idx % 10 == 0:
            self.get_logger().info(
                f'Waypoint {idx}: κ={current_curvature:.6f} m⁻¹, '
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
