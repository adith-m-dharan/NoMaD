
import os
import sys
import time
import io
import matplotlib.pyplot as plt

# ROS
from rclpy.clock import Clock, Duration
from sensor_msgs.msg import Image

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional

# models
from vint_train.models.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)  
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes() 
    ros_image.step = ros_image.width
    return ros_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def is_valid_ros2_bag(path):
    if not os.path.isdir(path):
        return False

    files = os.listdir(path)
    if len(files) != 2 or "metadata.yaml" not in files:
        return False

    return any(file.endswith(".db3") for file in files)

def split_list(input_list, split_factor):
        n = len(input_list)
        if split_factor <= 0 or split_factor > n:
            raise ValueError("Number of sublists must \
                             be between 1 and the length of the input list.")
        
        # Calculate the size of each chunk
        chunk_size = n // split_factor
        remainder = n % split_factor
        
        result = []
        start = 0
        
        for i in range(split_factor):
            end = start + chunk_size + (1 if i < remainder else 0)
            result.append(input_list[start:end])
            start = end
        
        return tuple(result)

class Rate:
    def __init__(self, hz, clock: Clock = None):
        self.__hz = hz
        self.__clock = clock
        self.__time = self.now()

    def now(self):
        if self.__clock is None:
            return time.monotonic()
        return self.__clock.now().nanoseconds * 1e-9

    def sleep(self):
        sleep_duration = max(
            0,
            (1/self.__hz) - (self.now() - self.__time)
        )
        if self.__clock is None:
            time.sleep(sleep_duration)
        else:
            self.__clock.sleep_for(Duration(seconds=sleep_duration))
        self.__time = self.now()
