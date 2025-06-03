import rclpy
from seem_ros.ros2_wrapper.seem_wrapper import SEEMLifecycleNode

def main(args=None):
    rclpy.init(args=args)
    node = SEEMLifecycleNode()
    rclpy.spin(node)
    rclpy.shutdown()
