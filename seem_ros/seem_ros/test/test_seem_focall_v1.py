import os
import torch
import numpy as np
import cv2
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory

from seem_ros.utils.constants import COCO_PANOPTIC_CLASSES
from seem_ros.modeling.BaseModel import BaseModel
from seem_ros.modeling import build_model
from seem_ros.utils.arguments import load_opt_from_config_files
from seem_ros.utils.distributed import init_distributed
from seem_ros.demo.seem.tasks import interactive_infer_image


def get_config_path():
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'configs', 'seem', 'focall_unicl_lang_demo.yaml')


def get_weights_path():
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'seem_focall_v1.pt')


def load_model():
    """Load the SEEM model and initialize text embeddings."""
    opt = load_opt_from_config_files([get_config_path()])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(get_weights_path()).eval().cuda()

    # Preload COCO text embeddings for grounding
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
        )

    return model


def prepare_input(image_path):
    """Load image and return it along with a dummy mask."""
    pil_image = PILImage.open(image_path).convert("RGB")
    mask = PILImage.new("RGB", pil_image.size, (0, 0, 0))  # Dummy black mask

    return {
        "image": pil_image,
        "mask": mask
    }


@torch.no_grad()
def run_text_inference(model, image_dict, query):
    result_image, cosine_sim = interactive_infer_image(
        model=model,
        audio_model=None,
        image=image_dict,
        tasks=["Text"],
        reftxt=query
    )
    return result_image, cosine_sim


@torch.no_grad()
def run_panoptic_inference(model, image_dict):
    result_image, _ = interactive_infer_image(
        model=model,
        audio_model=None,
        image=image_dict,
        tasks=["Panoptic"]
    )
    return result_image


@torch.no_grad()
def segment_image_by_reference(model, image_dict: dict, ref_mask: PILImage.Image) -> PILImage.Image:
    """Segment objects similar to a reference mask in the given image."""
    # Build input dicts
    input_dict = {
        "image": image_dict["image"],
        "mask": PILImage.new("RGB", image_dict["image"].size, (0, 0, 0))
    }
    ref_dict = {
        "image": image_dict["image"],
        "mask": ref_mask
    }

    # Run example-based segmentation
    result_image, cosine_sim = interactive_infer_image(
        model=model,
        audio_model=None,
        image=input_dict,
        tasks=["Example"],
        refimg=ref_dict
    )
    return result_image, cosine_sim


def show_result(pil_image):
    """Display the output image using OpenCV."""
    np_image = np.array(pil_image)
    cv2.imshow("SEEM Segmentation Result", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_mask_interactively(image: PILImage.Image, window_name="Draw Mask") -> PILImage.Image:
    """
    Opens an OpenCV window to draw a binary mask on the given image.
    Returns a PIL.Image (RGB) mask.
    """
    drawing = False
    radius = 10
    color = (255, 255, 255)  # White mask in RGB
    thickness = -1

    image_np = np.array(image.convert("RGB"))
    mask = np.zeros_like(image_np, dtype=np.uint8)

    def draw_circle(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, (x, y), radius, color, thickness)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(mask, (x, y), radius, color, thickness)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_circle)

    while True:
        preview = cv2.addWeighted(image_np, 0.7, mask, 0.3, 0)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 27:  # Enter or ESC to finish
            break

    cv2.destroyAllWindows()
    return PILImage.fromarray(mask)


def main():
    # File paths
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")
    image_ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image2.png")
    prompt = "zebra"

    print("Loading model...")
    model = load_model()

    print("Preparing input...")
    image_input = prepare_input(image_path)
    image_ref = prepare_input(image_ref_path)

    print("Running text inference with prompt:", prompt)
    output_image, cosine_sim = run_text_inference(model, image_input, prompt)
    if cosine_sim is not None:
        print("Cosine similarity:", cosine_sim)
    show_result(output_image)

    print("Running panoptic segmentation...")
    output_image = run_panoptic_inference(model, image_input)
    show_result(output_image)

    print("Drawing reference mask...")
    ref_mask = draw_mask_interactively(image_ref["image"])

    print("Running example-based segmentation...")
    output_image, cosine_sim = segment_image_by_reference(model, image_input, ref_mask)
    # if cosine_sim is not None:
    #     print("Cosine similarity:", cosine_sim)
    show_result(output_image)


if __name__ == "__main__":
    main()
