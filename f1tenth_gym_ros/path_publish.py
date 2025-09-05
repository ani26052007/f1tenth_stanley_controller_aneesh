#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(Path, 'waypoints_path', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.waypoints = self.read_waypoints('/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/spilberg_centerline.csv')
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')

    def read_waypoints(self, csv_filename):
        try:
            # FIXED: Load only first 2 columns (x, y) using usecols parameter
            data = np.loadtxt(csv_filename, delimiter=',', usecols=(0,1), comments='#')
            waypoints = [(x, y) for x, y in data]
            return waypoints
        except Exception as e:
            self.get_logger().error(f'Error loading CSV: {str(e)}')
            return []

    def timer_callback(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
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
