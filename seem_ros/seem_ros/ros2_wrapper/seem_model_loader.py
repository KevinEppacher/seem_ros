# seem_ros/core/seem_model_loader.py

import os
import rclpy
from seem_ros.utils.arguments import load_opt_from_config_files
from seem_ros.utils.distributed import init_distributed
from seem_ros.modeling import build_model
from seem_ros.modeling.BaseModel import BaseModel

def get_model(node, config_dir: str, weight_dir: str):
    config_name = node.get_parameter('config_name').get_parameter_value().string_value
    config_path = os.path.join(config_dir, config_name)

    if not os.path.exists(config_path):
        node.get_logger().error(f"Config file not found: {config_path}")
        return None

    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)

    weight_file = 'seem_focall_v1.pt' if 'focall' in config_name else 'seem_focall_v1.pt'
    weights_path = os.path.join(weight_dir, weight_file)

    if not os.path.exists(weights_path):
        node.get_logger().error(f"Weight file not found: {weights_path}")
        return None

    node.get_logger().info(f"SEEM model loaded with config: {config_name}")
    return BaseModel(opt, build_model(opt)).from_pretrained(weights_path).eval().cuda()
