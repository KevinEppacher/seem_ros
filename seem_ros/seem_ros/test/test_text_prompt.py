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
from torchvision import transforms
from types import SimpleNamespace


def get_config_path():
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'configs', 'seem', 'focall_unicl_lang_demo.yaml')


def get_weights_path():
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'seem_focall_v0.pt')


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

    model.model.metadata = SimpleNamespace(
        thing_dataset_id_to_contiguous_id={},
        stuff_dataset_id_to_contiguous_id={},
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


def run_inference(model, image_input, prompt):
    """Run SEEM inference with text-based grounding."""
    with torch.no_grad():
        result_image, cosine_sim = interactive_infer_image(
            model=model,
            audio_model=None,
            image=image_input,
            tasks=["Text"],
            reftxt=prompt
        )
    return result_image, cosine_sim 


def show_result(pil_image):
    """Display the output image using OpenCV."""
    np_image = np.array(pil_image)
    cv2.imshow("SEEM Segmentation Result", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(rgb, input_size=360):
    """Resize and convert to torch tensor"""
    transform = transforms.Resize(input_size, interpolation=PILImage.BICUBIC)
    image = np.asarray(transform(PILImage.fromarray(rgb)))
    tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
    return tensor

@torch.inference_mode()
def encode_image(model, rgb_np, mode="default"):
    """Encode an RGB image using SEEM-style input."""
    # Transform RGB (np.array) → Tensor
    transform = transforms.Compose([
        transforms.Resize(360, interpolation=PILImage.BICUBIC),
        transforms.ToTensor(),  # HWC → CHW, float in [0,1]
    ])
    pil_image = PILImage.fromarray(rgb_np)
    tensor = transform(pil_image).unsqueeze(0).cuda()  # (1, 3, H, W)

    # Wrap into model input dict
    input_dict = [{"image": tensor.squeeze(0)}]  # shape (3, H, W)
    return model(input_dict, mode)

@torch.inference_mode()
def encode_text(model, texts):
    return model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(texts, is_eval=True)

def encode_prompt(model, prompt, task="default"):
    if task == "default":
        return encode_text(model, prompt)

def main():
    prompt = "chair"
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")

    print("Loading model...")
    model = load_model()

    print("Preparing input...")
    image_input = prepare_input(image_path)

    pil_image = image_input["image"]
    rgb = np.array(pil_image)  # shape (H, W, 3)

    t_emb = encode_prompt(model=model, prompt=prompt, task="default")
    print("Text embedding:", t_emb)
    # print("Text embedding shape:", t_emb.shape)

    import time
    time.sleep(3)

    res_list = encode_image(model, rgb, mode="default")
    print("Encoded image:", res_list)
    # print("Encoded image shape:", res_list[0].shape)

if __name__ == "__main__":
    main()