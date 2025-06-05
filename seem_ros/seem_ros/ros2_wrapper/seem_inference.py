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
from seem_ros_interfaces.srv import Panoptic
from seem_ros.ros2_wrapper.utils import ros2_image_to_pil, pil_to_ros2_image


class SEEMInference:
    def __init__(self, node, config_dir, weight_dir):
        self.node = node
        self.model = get_model(node, config_dir, weight_dir)
        self.bridge = CvBridge()

        if self.model:
            with torch.no_grad():
                self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
                )

    def prepare_input(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        mask = PILImage.new("RGB", pil_image.size, (0, 0, 0))
        return {"image": pil_image, "mask": mask}

    @torch.no_grad()
    def run_text_inference(self, ros_image, query):
        image_input = self.prepare_input(ros_image)
        from seem_ros.demo.seem.tasks import interactive_infer_image
        result_image, cosine_sim = interactive_infer_image(
            model=self.model,
            audio_model=None,
            image=image_input,
            tasks=["Text"],
            reftxt=query
        )
        return result_image, cosine_sim

    @torch.no_grad()
    def run_panoptic_inference(self, ros_image):
        image_input = self.prepare_input(ros_image)
        from seem_ros.demo.seem.tasks import interactive_infer_image
        result_image, _ = interactive_infer_image(
            model=self.model,
            audio_model=None,
            image=image_input,
            tasks=["Panoptic"]
        )
        return result_image
