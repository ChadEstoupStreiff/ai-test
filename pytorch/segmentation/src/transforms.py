import torchvision.transforms.functional as TF
import torch
import logging

class CustomResize:
    def __init__(self, size):
        """
        Initialize the transformation with the desired output size.
        
        Args:
            size (tuple): Target size (height, width) for resizing.
        """
        self.size = size

    def __call__(self, data):
        """
        Resizes the image and adjusts the target accordingly.
        
        Args:
            image (PIL.Image): The input image.
            target (dict): Dictionary containing 'boxes' and optionally 'masks'.
        
        Returns:
            image (Tensor): Resized image as a tensor.
            target (dict): Updated target dictionary with resized 'boxes' and 'masks'.
        """
        image, target = data

        # Get original image dimensions
        original_width, original_height = image.size
        target_height, target_width = self.size

        # Resize image
        image = TF.resize(image, self.size)
        image = TF.to_tensor(image)

        # Scale bounding boxes
        boxes = target['boxes']
        x_scale = target_width / original_width
        y_scale = target_height / original_height

        # Update box coordinates (x1, y1, x2, y2)
        boxes[:, [0, 2]] *= x_scale  # Scale x-coordinates
        boxes[:, [1, 3]] *= y_scale  # Scale y-coordinates
        target['boxes'] = boxes

        # Resize masks if present
        if 'masks' in target:
            masks = target['masks']
            masks = TF.resize(masks.unsqueeze(0).float(), self.size).squeeze(0).byte()
            target['masks'] = masks

        return (image, target)
