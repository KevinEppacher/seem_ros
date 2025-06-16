import os
import torch
import numpy as np
import cv2
from PIL import Image
from ament_index_python.packages import get_package_share_directory

from seem_ros.utils.constants import COCO_PANOPTIC_CLASSES
from seem_ros.modeling.BaseModel import BaseModel
from seem_ros.modeling import build_model
from seem_ros.utils.arguments import load_opt_from_config_files
from seem_ros.utils.distributed import init_distributed
from seem_ros.demo.seem.tasks import interactive_infer_image
import torch.nn as nn
from torchvision import transforms
from ament_index_python.packages import get_package_share_directory


pkg_share = get_package_share_directory("seem_ros")

MODELS = {
    "seem": {
        "input_size": 360,
        "config": os.path.join(pkg_share, "configs", "seem", "seem_focall_lang.yaml"),
        "checkpoint": os.path.join(pkg_share, "seem_focall_v0.pt"),
    }
}

class VLFM(nn.Module):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.meta = MODELS[name]
        self.transform = transforms.Resize(
            kwargs.get("input_size", self.meta["input_size"]),
            interpolation=Image.BICUBIC
        )

    def encode_prompt(self, prompt, task="default"):
        if task == "default":
            return self.encode_text(prompt)

    def encode_text(self, text):
        pass

    def preprocess_image(self, rgb):
        pass

    def encode_image(self, image):
        pass

def get_config_path():
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'configs', 'seem', 'focall_unicl_lang_demo.yaml')


def get_weights_path():
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'seem_focall_v0.pt')


class RegionAlignedModel(VLFM):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(name, **kwargs)
        opt = load_opt_from_config_files([get_config_path()])
        opt = init_distributed(opt)
        model = BaseModel(opt, build_model(opt)).from_pretrained(get_weights_path()).eval().cuda()
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
            )

    @staticmethod
    def model_names():
        return ["seem"]

    @property
    def dim(self):
        return 512

    @torch.inference_mode()
    def encode_text(self, texts):
        return self.model.encode_text(texts)

    def preprocess_image(self, rgb):
        images = [np.asarray(self.transform(Image.fromarray(i))) for i in rgb]
        # NOTE: normalize image inside model
        images = torch.tensor(
            np.asarray(images, dtype=np.float32)
        ).float().permute(0, 3, 1, 2).cuda()
        return images

    @torch.inference_mode()
    def encode_image(self, rgb, mode="default"):
        rgb_images = self.preprocess_image(rgb)
        assert rgb_images.shape[1] == 3
        return self.model(rgb_images, mode)


def main():
    prompt = "chair"
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")

    model = RegionAlignedModel("seem", input_size=360)

    res_list = model.encode_image(image_path, mode="default")


if __name__ == "__main__":
    main()
