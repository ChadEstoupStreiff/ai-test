import cv2
import numpy as np
import logging
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import time
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
import torch.nn.functional as F_torch



NUM_CLASSES = 3 # Update based on your dataset (e.g., background + objects)
IMAGE_SIZE = (512, 512)  # Adjust if needed
SCORE_THRESHOLD = 0.9
MASK_THRESHOLD = 0.2  # Confidence threshold for masks
MODEL_PATH = "./maskrcnn_coco_model.pth"  # Path to your trained model

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Running on device: {device}")
logging.info("Loading trained model...")

model = maskrcnn_resnet50_fpn(pretrained=True)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
# in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# hidden_layer = 256
# model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)
logging.info("Model loaded successfully.")


# Function to preprocess frame
def preprocess_frame(frame):
    # Resize and normalize
    frame_resized = cv2.resize(frame, IMAGE_SIZE)
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    # Convert to tensor and permute dimensions to [C, H, W]
    frame_tensor = torch.tensor(frame_normalized, dtype=torch.float32, device=device).permute(2, 0, 1)
    # Add batch dimension [1, C, H, W]
    return frame_tensor.unsqueeze(0)

# Function to postprocess results
def postprocess_mask(mask, original_shape):
    # Convert boolean mask to float32 for processing
    mask_float = mask.astype(np.float32)  # Convert to float for scaling
    mask_resized = cv2.resize(mask_float, (original_shape[1], original_shape[0]))  # Resize to original frame size
    mask_uint8 = (mask_resized * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return mask_uint8


# Access camera
logging.info("Accessing camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change index for different cameras

# Configure logging

# Main loop with timing
while cap.isOpened():
    start_time = time.time()

    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break
    logging.info(f"\n\nFrame capture time: {time.time() - start_time:.4f} seconds")

    # Preprocess frame
    start_time = time.time()
    input_frame = preprocess_frame(frame)
    logging.info(f"Preprocessing time: {time.time() - start_time:.4f} seconds")

    # Perform inference
    start_time = time.time()
    with torch.no_grad():
        predictions = model(input_frame)[0]
    logging.info(f"Inference time: {time.time() - start_time:.4f} seconds")

    boxes = predictions.get('boxes', None)
    labels = predictions.get('labels', None)
    scores = predictions.get('scores', None)
    masks = predictions.get('masks', None)


    # Postprocess results
    start_time = time.time()

    # Resize masks to match the frame size
    if masks is not None:
        masks = F_torch.interpolate(
            masks, size=(frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False
        )
        masks = masks > MASK_THRESHOLD  # Binary threshold
        masks = masks.to(dtype=torch.bool)  # Convert to boolean
        filtered_masks = [mask for mask, score in zip(masks, scores) if score >= SCORE_THRESHOLD]
    else:
        filtered_masks = []

    # Draw masks
    frame_tensor = F.to_tensor(frame).mul(255).byte()
    if filtered_masks:
        frame_with_masks = draw_segmentation_masks(
            image=frame_tensor,
            masks=torch.stack(filtered_masks).squeeze(1),
            alpha=0.4
        )
    else:
        frame_with_masks = frame_tensor

    x_ratio = frame.shape[1] / IMAGE_SIZE[0]
    y_ratio = frame.shape[0] / IMAGE_SIZE[1]
    boxes = torch.tensor([[box[0] * x_ratio, box[1] * y_ratio, box[2] * x_ratio, box[3] * y_ratio] for box, score in zip(boxes, scores) if score >= SCORE_THRESHOLD])
    if boxes is not None and boxes.shape[0] > 0:
        filtered_labels = [f"{label.item()} : {score}" for label, score in zip(labels, scores) if score >= SCORE_THRESHOLD]

        # Draw bounding boxes
        frame_with_boxes = draw_bounding_boxes(
            image=frame_with_masks,
            boxes=boxes,
            labels=filtered_labels,
            colors=None,
            width=2
        )
    else:
        frame_with_boxes = frame_with_masks

    frame = frame_with_boxes.permute(1, 2, 0).cpu().numpy()

    # Display the frame
    cv2.imshow("Segmented Frame with Boxes", frame)


    logging.info(f"Drawing time: {time.time() - start_time:.4f} seconds")

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
