import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import torchvision.transforms as T
from pycocotools.coco import COCO
import os
import logging
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision.datasets.vision import VisionDataset
import json
from pycocotools import mask as coco_mask

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length bounding boxes and masks.

    Args:
        batch: A list of tuples (image, target), where
               - image is a Tensor of shape [C, H, W]
               - target is a dict containing 'boxes', 'labels', and optionally 'masks'

    Returns:
        images: List of image tensors.
        targets: List of target dictionaries.
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.transforms = transforms
        self.image_ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        masks = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            masks.append(self.coco.annToMask(ann))
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes),
            'labels': torch.tensor(labels),
            'masks': torch.tensor(masks),
            'image_id': torch.tensor([image_id])
        }

        if self.transforms:
            image, target = self.transforms((image, target))

        return image, target

    def __len__(self):
        return len(self.image_ids)
