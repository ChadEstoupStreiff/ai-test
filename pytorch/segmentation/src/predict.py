import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.utils import draw_segmentation_masks
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import os
import logging
import numpy as np

# Constants
NUM_CLASSES = 3  # Update based on your dataset (e.g., background + objects)
IMAGE_SIZE = (512, 512)  # Adjust if needed
DATASET_PATH = "../data/cell/prediction"  # Path to input dataset
OUTPUT_PATH = "../data/cell/output/predictions2"  # Path for output results
SCORE_THRESHOLD = 0.
MASK_THRESHOLD = 0.05  # Confidence threshold for masks
MODEL_PATH = "../data/cell/models/maskrcnn_coco_model.pth"  # Path to your trained model

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Running on device: {device}")
logging.info("Loading trained model...")

model = maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)
logging.info("Model loaded successfully.")

# Iterate over images in dataset
for image_name in os.listdir(DATASET_PATH):
    image_path = os.path.join(DATASET_PATH, image_name)
    logging.info(f"Processing image: {image_path}")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)  # Move to device

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)[0]  # Get outputs for the single image

    # Prepare image for drawing

    # Draw bounding boxes and labels
    boxes = outputs.get('boxes', [])
    labels = outputs.get('labels', [])
    scores = outputs.get('scores', [])
    masks = outputs.get('masks', None)

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)  # Path to a .ttf font file
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font
    
    logging.info(f"Number of boxes: {len(boxes)}")

    for box, label, score in zip(boxes, labels, scores):
        if score >= SCORE_THRESHOLD:  # Filter by score threshold
            x1, y1, x2, y2 = box.tolist()
            label_text = f"Class {label.item()} ({score:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 50), width=2)
            draw.text((x1, y1), label_text, fill="red", font=font)

    # Extract masks and overlay them
    if masks is not None:
        masks = masks > MASK_THRESHOLD  # Convert to binary masks
        filtered_masks = []
        for mask, score in zip(masks, scores):
            if score >= SCORE_THRESHOLD:  # Filter masks by score threshold
                filtered_masks.append(mask)
        if filtered_masks:
            image_with_masks = draw_segmentation_masks(
                image=F.to_tensor(draw_image).mul(255).byte(),
                masks=torch.stack(filtered_masks).squeeze(1),
                alpha=0.4
            )
            final_image = Image.fromarray(image_with_masks.permute(1, 2, 0).cpu().numpy())
        else:
            final_image = draw_image
    else:
        final_image = draw_image

    # Save the result
    output_file_path = os.path.join(OUTPUT_PATH, f"{os.path.splitext(image_name)[0]}_segmentation.png")
    final_image.save(output_file_path)
    logging.info(f"Saved segmentation result to {output_file_path}")