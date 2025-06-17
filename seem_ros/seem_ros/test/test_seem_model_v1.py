import os
import cv2
import torch
import numpy as np
from detectron2.data.detection_utils import read_image
from detectron2.structures import ImageList
import torch
import numpy as np

from seem_ros.modeling.architectures import GeneralizedSEEM
from seem_ros.utils.constants import COCO_PANOPTIC_CLASSES
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.colormap import random_color
import yaml

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_config_path():
    from ament_index_python.packages import get_package_share_directory
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'configs', 'seem', 'focall_unicl_lang_demo.yaml')

def get_weights_path():
    from ament_index_python.packages import get_package_share_directory
    pkg_share = get_package_share_directory('seem_ros')
    return os.path.join(pkg_share, 'seem_focall_v0.pt')

def to_image_tensor(img):
    """
    Converts a numpy image (HWC, BGR or RGB, 0–255) to a torch tensor (CHW, float32, 0–1).
    Fixes negative strides by copying the array.
    """
    if isinstance(img, np.ndarray):
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW + copy if needed
    return torch.from_numpy(img).float() / 255.0

def main():
    # Prompt + Bild vorbereiten
    prompt = ["chair"]
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")
    image_bgr = read_image(image_path, format="BGR")
    height, width = image_bgr.shape[:2]
    image_tensor = to_image_tensor(image_bgr).unsqueeze(0).cuda()

    # Modell laden
    cfg_dict = load_yaml(get_config_path())
    model_args = GeneralizedSEEM.from_config(cfg_dict)
    model = GeneralizedSEEM(**model_args).cuda().eval()

    from detectron2.data import MetadataCatalog
    model.metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    model.load_state_dict(torch.load(get_weights_path(), map_location="cuda"), strict=False)

    # Bild normalisieren & wrappen
    image = (image_tensor[0] - model.pixel_mean) / model.pixel_std
    images = ImageList.from_tensors([image], model.size_divisibility).to(model.device)

    with torch.no_grad():
        # Inferenz vorbereiten
        features = model.backbone(images.tensor)
        model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(prompt, name='default')
        outputs = model.sem_seg_head(features)

        mask_cls = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"][0]
        box_pred = outputs.get("pred_boxes", None)

        # === Semantic Segmentation ===
        print("\nSemantic Inference:")
        semseg = model.semantic_inference(mask_cls, mask_pred)
        print(" → Output shape:", semseg.shape)

        # === Panoptic Segmentation ===
        print("\nPanoptic Inference:")
        panoptic_seg, segments_info = model.panoptic_inference(mask_cls, mask_pred)
        print(" → Unique segment IDs:", torch.unique(panoptic_seg))
        print(" → Segments:", segments_info)

        # Resize panoptic mask to original image size
        panoptic_seg_resized = torch.nn.functional.interpolate(
            panoptic_seg.unsqueeze(0).unsqueeze(0).float(),
            size=(height, width),
            mode="nearest"
        )[0, 0].to(torch.int32)

        pano_np = panoptic_seg_resized.cpu().numpy().astype(np.uint8)


        # === Instance Segmentation ===
        print("\nInstance Inference:")
        num_valid = mask_cls.shape[0] * (mask_cls.shape[1] - 1)
        model.test_topk_per_image = min(model.test_topk_per_image, num_valid)
        instances = model.instance_inference(mask_cls, mask_pred, box_pred[0] if box_pred else None)
        print(f" → Found {len(instances)} instances.")

        for i in range(len(instances)):
            print(f"   - Class {instances.pred_classes[i].item()}, Score {instances.scores[i].item():.2f}")

        # === Visualisierung ===
        vis_img = image_bgr.copy()

        # Semantic overlay
        if semseg.shape[0] > 0:
            semseg_np = semseg.argmax(0).cpu().numpy().astype(np.uint8)
            sem_color = cv2.applyColorMap((semseg_np * 10).astype(np.uint8), cv2.COLORMAP_JET)
            sem_overlay = cv2.addWeighted(vis_img, 0.6, sem_color, 0.4, 0)
            cv2.imshow("Semantic Segmentation", sem_overlay)

        # Panoptic overlay
        if panoptic_seg is not None:
            pano_np = panoptic_seg.cpu().numpy().astype(np.uint8)
            pano_color = np.zeros_like(vis_img)
            for seg in segments_info:
                mask = pano_np == seg["id"]
                color = random_color(rgb=False)
                pano_color[mask] = color
            pano_overlay = cv2.addWeighted(vis_img, 0.6, pano_color, 0.4, 0)
            cv2.imshow("Panoptic Segmentation", pano_overlay)

        # Instance overlay
        if len(instances) > 0:
            instance_img = vis_img.copy()
            for i in range(len(instances)):
                mask = instances.pred_masks[i].cpu().numpy().astype(np.uint8)
                color = random_color(rgb=False)
                instance_img[mask > 0] = (0.6 * instance_img[mask > 0] + 0.4 * np.array(color)).astype(np.uint8)
                box = instances.pred_boxes.tensor[i].cpu().numpy().astype(int)
                class_id = instances.pred_classes[i].item()
                score = instances.scores[i].item()
                label = f"{class_id}: {score:.2f}"
                cv2.rectangle(instance_img, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(instance_img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow("Instance Segmentation", instance_img)

        # Original anzeigen
        cv2.imshow("Original", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
