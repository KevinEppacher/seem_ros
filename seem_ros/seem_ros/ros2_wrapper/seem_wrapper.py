from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import State
from lifecycle_msgs.msg import Transition
import rclpy
from rclpy.lifecycle import TransitionCallbackReturn
from rcl_interfaces.msg import ParameterDescriptor
import os
from ament_index_python.packages import get_package_share_directory
import torch
from seem_ros.utils.constants import COCO_PANOPTIC_CLASSES
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from seem_ros.demo.seem.tasks import interactive_infer_image
from PIL import Image as PILImage
import torch
import numpy as np
import cv2
# Custom import
from seem_ros.modeling.BaseModel import BaseModel
from seem_ros.modeling import build_model
from seem_ros.utils.arguments import load_opt_from_config_files
from seem_ros.utils.distributed import init_distributed
from seem_ros.ros2_wrapper.seem_model_loader import get_model
from seem_ros_interfaces.srv import Panoptic, ObjectSegmentation, SemanticSimilarity
from seem_ros.ros2_wrapper.utils import ros2_image_to_pil, pil_to_ros2_image
from seem_ros.ros2_wrapper.seem_inference import SEEMInference
from seem_ros.ros2_wrapper.service_handler import SEEMServiceHandler



class SEEMLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('seem_lifecycle_node')
        self.get_logger().info("Initializing SEEM Lifecycle Node...")

        self._declare_parameters()
        self.bridge = CvBridge()
        self.image_sub = None
        self.panoptic_pub = None
        self.timer = None
        self.rgb_image = None

        config_name = self.get_parameter('config_name').get_parameter_value().string_value
        pkg_share = get_package_share_directory('seem_ros')
        self.config_dir = os.path.join(pkg_share, 'configs/seem')
        self.weight_dir = pkg_share

        self.get_logger().info("Finished initializing SEEMLifecycleNode.")


    def _declare_parameters(self):
        self.declare_parameter('config_name', 'focall_unicl_lang_demo.yaml', ParameterDescriptor(description='Config file name'))
        self.declare_parameter('timer_frequency', 10.0, ParameterDescriptor(description='Timer frequency'))
        self.timer_frequency = self.get_parameter("timer_frequency").get_parameter_value().double_value

    def on_configure(self, state):
        self.inference = SEEMInference(self, self.config_dir, self.weight_dir)

        if self.inference.model:
            return TransitionCallbackReturn.SUCCESS
        return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State):
        try:
            self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)
            self.panoptic_pub = self.create_publisher(Image, '/panoptic', 1)
            self.timer = self.create_timer(1.0 / self.timer_frequency, self.timer_callback)

            self.service_handler = SEEMServiceHandler(self, self.inference)

            self.panoptic_srv = self.create_service(Panoptic, 'panoptic_segmentation', self.service_handler.panoptic)
            self.object_segmentation_srv = self.create_service(ObjectSegmentation, 'object_segmentation', self.service_handler.object_segmentation)
            self.semantic_similarity_srv = self.create_service(SemanticSimilarity, 'semantic_similarity', self.service_handler.semantic_similarity)

            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Activation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State):
        if self.timer:
            self.timer.cancel()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State):
        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, msg: Image):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image processing failed: {e}')

    def timer_callback(self):
        if self.rgb_image is None:
            return
