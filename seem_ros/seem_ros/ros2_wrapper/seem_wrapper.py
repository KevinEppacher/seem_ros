from vlm_base.vlm_base import VLMBaseLifecycleNode
from seem_ros_interfaces.srv import Panoptic, ObjectSegmentation, SemanticSimilarity
from seem_ros.ros2_wrapper.seem_inference import SEEMInference
from seem_ros.ros2_wrapper.service_handler import SEEMServiceHandler
from ament_index_python.packages import get_package_share_directory
from rclpy.lifecycle import TransitionCallbackReturn, State
from sensor_msgs.msg import Image
import os

class SEEMNode(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__('seem_lifecycle_node')
        self._declare_parameters()

        config_name = self.get_parameter('config_name').get_parameter_value().string_value
        pkg_share = get_package_share_directory('seem_ros')
        self.config_dir = os.path.join(pkg_share, 'configs/seem')
        self.weight_dir = pkg_share

    def _declare_parameters(self):
        self.declare_parameter('config_name', 'focall_unicl_lang_demo.yaml')
        self.declare_parameter('timer_frequency', 10.0)
        self.timer_frequency = self.get_parameter("timer_frequency").get_parameter_value().double_value

    def load_model(self):
        self.inference = SEEMInference(self, self.config_dir, self.weight_dir)
        return self.inference.model

    def create_services(self):
        self.service_handler = SEEMServiceHandler(self, self.inference)

        self.create_service(Panoptic, 'panoptic_segmentation', self.service_handler.panoptic)
        self.create_service(ObjectSegmentation, 'object_segmentation', self.service_handler.object_segmentation)
        self.create_service(SemanticSimilarity, 'semantic_similarity', self.service_handler.semantic_similarity)

        self.panoptic_pub = self.create_publisher(Image, '/panoptic', 1)
        self.timer = self.create_timer(1.0 / self.timer_frequency, self.timer_callback)

    def timer_callback(self):
        if self.rgb_image is None:
            return
        # Optional logic can go here, e.g. publishing panoptic result
