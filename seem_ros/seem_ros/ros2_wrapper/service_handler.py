from rclpy.lifecycle import LifecycleNode
from seem_ros.ros2_wrapper.seem_inference import SEEMInference
from seem_ros.ros2_wrapper.utils import ros2_image_to_pil, pil_to_ros2_image

class SEEMServiceHandler:
    def __init__(self, node: LifecycleNode, inference: SEEMInference):
        self.node = node
        self.inference = inference

    def panoptic(self, request, response):
        try:
            result = self.inference.run_panoptic_inference(request.image)
            if result:
                response.panoptic_segmentation = pil_to_ros2_image(result, frame_id="map", stamp=self.node.get_clock().now().to_msg())
            return response
        except Exception as e:
            self.node.get_logger().error(f"Panoptic service failed: {e}")
            return response

    def object_segmentation(self, request, response):
        try:
            result, _ = self.inference.run_text_inference(request.image, request.query)
            if result:
                response.segmented_image = pil_to_ros2_image(result, frame_id="map", stamp=self.node.get_clock().now().to_msg())
            return response
        except Exception as e:
            self.node.get_logger().error(f"Object segmentation failed: {e}")
            return response

    def semantic_similarity(self, request, response):
        try:
            _, score = self.inference.run_text_inference(request.image, request.query)
            response.score = float(score) if score is not None else float('nan')
            return response
        except Exception as e:
            self.node.get_logger().error(f"Semantic similarity failed: {e}")
            response.score = float('nan')
            return response

