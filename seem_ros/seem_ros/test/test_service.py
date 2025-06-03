import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from seem_ros_interfaces.srv import Panoptic

class SimplePanopticService(Node):
    def __init__(self):
        super().__init__('simple_panoptic_service_node')
        self.srv = self.create_service(Panoptic, 'panoptic_segmentation', self.handle_request)
        self.get_logger().info("SimplePanopticService is ready and waiting for requests.")

    def handle_request(self, request, response):
        self.get_logger().info("Received Panoptic service request.")
        response.panoptic_segmentation = request.image
        self.get_logger().info("Sending back the same image as segmentation.")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = SimplePanopticService()
    rclpy.spin(node)
    rclpy.shutdown()
