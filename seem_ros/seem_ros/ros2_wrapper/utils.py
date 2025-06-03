from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import cv2
import numpy as np
import rclpy

def ros2_image_to_pil(ros_image: Image, logger=None) -> PILImage.Image | None:
    bridge = CvBridge()
    try:
        # Convert ROS2 Image to OpenCV format (BGR)
        cv_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = PILImage.fromarray(rgb_image)
        return pil_image

    except Exception as e:
        msg = f"Failed to convert ROS2 image to PIL: {e}"
        if logger:
            logger.error(msg)
        else:
            print("[ERROR]", msg)
        return None
    
def pil_to_ros2_image(pil_image: PILImage, frame_id="map", stamp=None):
    bridge = CvBridge()
    np_image = np.array(pil_image)
    ros_image = bridge.cv2_to_imgmsg(np_image, encoding="rgb8")
    ros_image.header.frame_id = frame_id
    if stamp is not None:
        ros_image.header.stamp = stamp
    return ros_image
