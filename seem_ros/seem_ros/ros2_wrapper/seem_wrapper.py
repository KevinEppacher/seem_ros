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
from seem_ros.ros2_wrapper.seem_inference import run_text_inference, prepare_input, run_panoptic_inference

class SEEMLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('seem_lifecycle_node')
        self.get_logger().info("Initializing SEEM Lifecycle Node...")

        self.declare_parameter('config_name','focall_unicl_lang_demo.yaml',ParameterDescriptor(description='Name of the config file to use.'))
        self.declare_parameter('timer_frequency', 10.0, ParameterDescriptor(description='Frequency of the timer.'))

        self.timer_frequency = self.get_parameter("timer_frequency").get_parameter_value().double_value


        pkg_share = get_package_share_directory('seem_ros')
        self.config_dir = os.path.join(pkg_share, 'configs/seem')

        self.weight_dir = pkg_share

        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = None

        # Publishers
        self.panoptic_pub = None
        
        # Timers
        self.timer = None

        # Service
        self.panoptic_srv = None

        self.model = None
        self.rgb_image = None

    def on_configure(self, state):
        rclpy.logging.get_logger('seem_lifecycle_node').info('Configuring SEEM Lifecycle Node...')
        
        try:
            self.model = get_model(self, self.config_dir, self.weight_dir)

            with torch.no_grad():
                self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            rclpy.logging.get_logger('seem_lifecycle_node').error(f'Configuration failed: {e}')
            return TransitionCallbackReturn.FAILURE
             
    def on_activate(self, state: State):
        rclpy.logging.get_logger('seem_lifecycle_node').info('Activating...')
        try:
            # Subscribers
            self.image_sub = self.create_subscription(Image,'/rgb',self.image_callback,10)

            # Publishers
            self.panoptic_pub = self.create_publisher(Image, '/panoptic', 1)

            # Timers
            self.createTimer()

            # Service
            self.panoptic_srv = self.create_service(Panoptic,'panoptic_segmentation',self.handle_panoptic_request)

            self.object_segmentation_srv = self.create_service(ObjectSegmentation,'object_segmentation',self.handle_object_segmentation_request)

            self.semantic_similarity_srv = self.create_service(SemanticSimilarity,'semantic_similarity',self.handle_semantic_similarity_request)

            rclpy.logging.get_logger('seem_lifecycle_node').info('SEEM activated.')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            rclpy.logging.get_logger('seem_lifecycle_node').error(f'Activation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State):
        rclpy.logging.get_logger('value_map_node').info('Deactivating...')
        try:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
                rclpy.logging.get_logger('value_map_node').info("Timer cancelled.")

            if self.panoptic_pub is not None:
                self.panoptic_pub = None

            if self.panoptic_srv is not None:
                self.panoptic_srv.destroy()
                self.panoptic_srv = None

            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            rclpy.logging.get_logger('value_map_node').error(f'Deactivation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: State):
        self.get_logger().info('Cleaning up...')
        try:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Cleanup failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: State):
        rclpy.logging.get_logger('seem_lifecycle_node').info('Shutting down...')
        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, msg: Image):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rclpy.logging.get_logger('seem_lifecycle_node').error(f'Image processing failed: {e}')

    def createTimer(self):
        self.timer = self.create_timer(1.0 / self.timer_frequency, self.timer_callback)
        self.get_logger().info(f"Timer started with frequency: {self.timer_frequency} Hz")

    def timer_callback(self):
        if self.rgb_image is None and self.model is None:
            rclpy.logging.get_logger('seem_lifecycle_node').warn("No image received or model not loaded.")
            return

        # try:
        #     with torch.no_grad():
        #         output_image = self.run_panoptic_inference(self.model, self.rgb_image)

        #         if isinstance(output_image, PILImage.Image):
        #             msg = self.bridge.cv2_to_imgmsg(np.array(output_image), encoding='rgb8')
        #             msg.header.stamp = self.get_clock().now().to_msg()
        #             msg.header.frame_id = "map"
        #             self.panoptic_pub.publish(msg)
        #             self.get_logger().info("Published panoptic segmentation.")
        # except Exception as e:
        #     self.get_logger().error(f"Panoptic inference failed: {e}")
    
    @torch.no_grad()
    def run_panoptic_inference(self, model, ros_image: Image) -> Image:
        try:
            if ros_image is None:
                self.get_logger().error("Received empty image in Panoptic inference.")
                return None
            
            image_input = prepare_input(self, ros_image)

            result_image = run_panoptic_inference(self, model, image_input)

            if not isinstance(result_image, PILImage.Image):
                self.get_logger().error("Output is not a PIL.Image!")
                return None

            return pil_to_ros2_image(result_image, frame_id="map", stamp=self.get_clock().now().to_msg())

        except Exception as e:
            self.get_logger().error(f"run_panoptic_inference failed: {e}")
            return None

    def handle_panoptic_request(self, request, response):
        if request.image is None:
            self.get_logger().error("Received empty image in Panoptic service request.")
            return response

        panoptic_image = self.run_panoptic_inference(self.model, request.image)

        if panoptic_image is None:
            self.get_logger().error("Panoptic inference returned None.")
            return response

        response.panoptic_segmentation = panoptic_image
        self.get_logger().info("Returning panoptic segmentation image.")
        return response

    def handle_object_segmentation_request(self, request, response):
        if request.image is None:
            self.get_logger().error("Received empty image in Object Segmentation service request.")
            return response

        image_input = prepare_input(self, request.image)
        object_segmentation_image, _ = run_text_inference(self, self.model, image_input, request.query)

        if object_segmentation_image is None:
            self.get_logger().error("Object segmentation inference returned None.")
            return response

        response.segmented_image = pil_to_ros2_image(object_segmentation_image, frame_id="map", stamp=self.get_clock().now().to_msg())
        # response.segmented_image = object_segmentation_image
        self.get_logger().info("Returning object segmentation image.")
        return response
    
    def handle_semantic_similarity_request(self, request, response):
        if request.image is None or request.query.strip() == "":
            self.get_logger().error("Empty image or query in SemanticSimilarity request.")
            response.score = float('nan')
            return response

        try:
            image_input = prepare_input(self, request.image)
            _, cosine_sim = run_text_inference(self, self.model, image_input, request.query)

            if cosine_sim is None:
                self.get_logger().error("Cosine similarity is None.")
                response.score = float('nan')
            else:
                response.score = float(cosine_sim)

            self.get_logger().info(f"Returning cosine similarity: {response.score:.4f}")
            return response

        except Exception as e:
            self.get_logger().error(f"Semantic similarity computation failed: {e}")
            response.score = float('nan')
            return response
