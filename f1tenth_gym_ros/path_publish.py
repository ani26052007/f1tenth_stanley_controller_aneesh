#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import csv
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(Path, 'waypoints_path', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Load waypoints from CSV
        self.waypoints = self.read_waypoints('/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/csv_spielberg_map.csv')
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')

    def read_waypoints(self, csv_filename):
        waypoints = []
        try:
            with open(csv_filename, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                for row in reader:
                    # Skip header or comments
                    if not row or row[0].strip().startswith('#'):
                        continue
                    x = float(row[0].strip())
                    y = float(row[1].strip())
                    waypoints.append((x, y))
        except FileNotFoundError:
            self.get_logger().error(f'CSV file {csv_filename} not found!')
        except Exception as e:
            self.get_logger().error(f'Error reading CSV: {str(e)}')
        return waypoints

    def timer_callback(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation
            path_msg.poses.append(pose)
        
        self.publisher_.publish(path_msg)
        self.get_logger().info(f'Published path with {len(self.waypoints)} waypoints')

def main(args=None):
    rclpy.init(args=args)
    path_publisher = PathPublisher()
    try:
        rclpy.spin(path_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        path_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
