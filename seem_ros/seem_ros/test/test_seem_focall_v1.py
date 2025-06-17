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
def run_text_inference(model, ros_image, query):
    image_input = ros_image
    result_image, cosine_sim = interactive_infer_image(
        model=model,
        audio_model=None,
        image=image_input,
        tasks=["Text"],
        reftxt=query
    )
    return result_image, cosine_sim

@torch.no_grad()
def run_panoptic_inference(model, ros_image):
    image_input = ros_image
    from seem_ros.demo.seem.tasks import interactive_infer_image
    result_image, _ = interactive_infer_image(
        model=model,
        audio_model=None,
        image=image_input,
        tasks=["Panoptic"]
    )
    return result_image


def show_result(pil_image):
    """Display the output image using OpenCV."""
    np_image = np.array(pil_image)
    cv2.imshow("SEEM Segmentation Result", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segment_image_by_reference(model, ros_image_dict, ref_image_dict) -> PILImage.Image:
    ref_image = ref_image_dict["image"]
    white_mask = PILImage.new("RGB", ref_image.size, (255, 255, 255))

    ref_dict = {
        "image": ref_image.convert("RGB"),
        "mask": white_mask
    }

    target_image = ros_image_dict["image"]
    target_dict = {
        "image": target_image.convert("RGB"),
        "mask": PILImage.new("RGB", target_image.size, (0, 0, 0))  # dummy mask
    }

    result_image, _ = interactive_infer_image(
        model=model,
        audio_model=None,
        image=target_dict,
        tasks=["Example"],
        refimg=ref_dict
    )

    return result_image

def prepare_reference_image(image_path):
    image = PILImage.open(image_path).convert("RGB")
    white_mask = PILImage.new("RGB", image.size, (255, 255, 255))  # white full mask
    return {
        "image": image,
        "mask": white_mask
    }



def main():
    prompt = "chair"
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ducks.png")
    image_ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "duck_input.png")

    print("Loading model...")
    
    model = load_model()

    print("Preparing input...")
    
    image_input = prepare_input(image_path)
    
    ref_image = prepare_reference_image(image_ref_path)    


    # print("Running inference with prompt:", prompt)
    # output_image, cosine_sim = run_text_inference(model, image_input, prompt)
    # if cosine_sim is not None:
    #     print("Cosine similarity:", cosine_sim)

    # print("Displaying result...")
    # show_result(output_image)

    # output_image = run_panoptic_inference(model, image_input)
    # show_result(output_image)

    show_result(ref_image["image"])  # zeigt das Referenzbild
    # oder
    show_result(ref_image["mask"])   # zeigt die Maske (meist schwarz mit weißem Objekt)
    output_image = segment_image_by_reference(model, image_input, ref_image)
    show_result(output_image)

if __name__ == "__main__":
    main()
