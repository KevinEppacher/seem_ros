import os
import torch
import numpy as np
import cv2
from PIL import Image as PILImage

from seem_ros.utils.constants import COCO_PANOPTIC_CLASSES
from seem_ros.modeling.BaseModel import BaseModel
from seem_ros.modeling import build_model
from seem_ros.utils.arguments import load_opt_from_config_files
from seem_ros.utils.distributed import init_distributed
from seem_ros.demo.seem.tasks import interactive_infer_image
from ament_index_python.packages import get_package_share_directory
import os


def get_config_path():
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'configs', 'seem', 'focall_unicl_lang_demo.yaml')

def get_weights_path():
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'seem_focall_v0.pt')

# --- Configuration ---
CONFIG_PATH = get_config_path()
WEIGHTS_PATH = get_weights_path()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_IMAGE_PATH = os.path.join(CURRENT_DIR, "image.png")

def load_model():
    opt = load_opt_from_config_files([CONFIG_PATH])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(WEIGHTS_PATH).eval().cuda()

    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
        )

    return model


def prepare_input(image_path, prompt="chair"):
    print("Preparing input...")
    pil_image = PILImage.open(image_path).convert("RGB")
    mask = PILImage.new("RGB", pil_image.size, (0, 0, 0))

    image_input = {
        "image": pil_image,
        "mask": mask,
        "text": [prompt]
    }

    return image_input


def run_inference(model, image_input, prompt="chair"):
    with torch.no_grad():
        result_image, _ = interactive_infer_image(
            model=model,
            audio_model=None,
            image=image_input,
            tasks=["Text"],
            reftxt=prompt
        )
    return result_image



def show_result(pil_image):
    np_image = np.array(pil_image)
    cv2.imshow("SEEM Segmentation Result", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("Loading model...")
    model = load_model()

    print("Preparing input...")
    image_input = prepare_input(EXAMPLE_IMAGE_PATH, prompt="chair")

    print("Running inference...")
    output_image = run_inference(model, image_input, prompt="chair")

    print("Displaying result...")
    show_result(output_image)


if __name__ == "__main__":
    main()
