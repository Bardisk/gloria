import json
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from .data_aug import RandomAugmentation

def build_transform(dataset_type, cfg):
    lower_type = dataset_type.lower()
    interpolation_method = InterpolationMode[cfg.INTERPOLATION.upper()]
    transform = None
    if cfg.NO_TRANSFORM:
        return transform
    if "train" in lower_type:
        transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(img_size)
                transforms.Resize(cfg.SIZE, interpolation=interpolation_method),
                RandomAugmentation(cfg, interpolation_method),
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(cfg.PIXEL_MEAN),
                    tuple(cfg.PIXEL_STD),
                ),
            ]
        )
    elif "val" in lower_type or "test" in lower_type:
        transform = transforms.Compose(
            [
                transforms.Resize(cfg.SIZE, interpolation=interpolation_method),
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(cfg.PIXEL_MEAN),
                    tuple(cfg.PIXEL_STD),
                ),
            ]
        )
    return transform

class general_wrapper(data.dataset):
    def __init__(
        self,
        path,   # path of the dataset
        dataset_split,
    )