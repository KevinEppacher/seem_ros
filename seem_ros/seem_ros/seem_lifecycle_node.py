import rclpy
from seem_ros.ros2_wrapper.seem_wrapper import SEEMNode

def main(args=None):
    rclpy.init(args=args)
    node = SEEMNode()
    rclpy.spin(node)
    rclpy.shutdown()
