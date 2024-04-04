import json
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from yacs.config import CfgNode

# relative import
from data_aug import RandomAugmentation

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

class CXR_dataset(data.Dataset):

    def __init__(
        self,
        name,
        ds_type="test",
        root="datasets",  # path of "datasets"
        proportion=1
    ):
        self.name = name
        self.ds_type = ds_type
        # turn the path into the specified dataset
        self.root = root = os.path.join(root, name)
        # load classnames from text
        with open(os.path.join(root, "classnames.txt"), "r", encoding="utf-8") as f:
            self.classnames = eval(f.read())
        # print(self.classnames)

        # cfg.INPUT defaults
        self.cfg = CfgNode()
        self.cfg.INPUT = CfgNode()
        self.cfg.INPUT.SIZE = (224, 224)
        self.cfg.INPUT.INTERPOLATION = "bicubic"
        self.cfg.INPUT.NO_TRANSFORM = False
        self.cfg.INPUT.CROP_SCALE = (0.8, 1.0)
        self.cfg.INPUT.ROTATE_DEGREES = (-10, 10)
        self.cfg.INPUT.AFFINE_DEGREES = (-10, 10)
        self.cfg.INPUT.AFFINE_TRANSLATE = (0.0625, 0.0625)
        self.cfg.INPUT.AFFINE_SCALE = (0.9, 1.1)
        self.cfg.INPUT.HORIZONTAL_FLIP_PROB = 0.5
        self.cfg.INPUT.COLOR_JITTER_BRIGHTNESS = (0.8, 1.2)
        self.cfg.INPUT.COLOR_JITTER_CONTRAST = (0.8, 1.2)
        self.cfg.INPUT.PIXEL_MEAN = None
        self.cfg.INPUT.PIXEL_STD = None
        self.cfg.freeze()
        
        self.cfg.merge_from_file(os.path.join(root, "info.yaml"))

        if self.cfg.INPUT.PIXEL_MEAN is None or self.cfg.INPUT.PIXEL_STD is None:
            raise ValueError('PIXEL_MEAN or PIXEL_STD not set! We have no defaults for them.')

        with open(os.path.join(root, f"{ds_type}.jsonl"), "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

        if self.ds_type == 'train':
            np.random.shuffle(self.data)
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            pass

        self.transform = build_transform(self.ds_type, self.cfg.INPUT)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, f"{self.ds_type}_dataset", self.data[index]["filepath"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        text = self.data[index]["text"]
        labels = torch.Tensor(self.data[index]["labels"])
        target = labels

        return img, text, target

# relative import, this module cant be executed as main module.
if __name__ == '__main__':
   dataset = CXR_dataset("vindr")

