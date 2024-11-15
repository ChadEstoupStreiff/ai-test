import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import torchvision.transforms as T
from pycocotools.coco import COCO
import os
import logging
from PIL import Image
from tqdm import tqdm
from dataset import CocoDataset, custom_collate_fn
from transforms import CustomResize



# Constants
DATASET_PATH = "../data/cell"
TRAIN_ANNOTATIONS = os.path.join(DATASET_PATH, "result.json")
VAL_ANNOTATIONS = os.path.join(DATASET_PATH, "result.json")
MODEL_SAVE_PATH = "../data/cell/models/"
MODEL_NAME = "maskrcnn_coco_model.pth"
NUM_CLASSES = 3  # Update based on your dataset (e.g., background + objects)
EPOCHS = 200
BATCH_SIZE = 4
IMAGE_SIZE = (512, 512)
LEARNING_RATE = 0.005

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
os.makedirs(MODEL_SAVE_PATH)

# Load datasets
logging.info("Loading datasets...")
train_transforms= transforms.Compose([CustomResize(IMAGE_SIZE)])
val_transforms= transforms.Compose([CustomResize(IMAGE_SIZE)])

train_dataset = CocoDataset(DATASET_PATH, TRAIN_ANNOTATIONS, transforms=train_transforms)
val_dataset = CocoDataset(DATASET_PATH, VAL_ANNOTATIONS, transforms=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# Initialize model, optimizer, and scheduler
logging.info("Initializing model...")
model = maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
logging.info("Starting training...")

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    # Training phase
    model.train()
    train_loss = 0
    for images, targets in tqdm(train_loader, desc="Training"):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        for loss_name, loss_value in loss_dict.items():
            logging.info(f"{loss_name}: {loss_value.item()}")
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        train_loss += losses.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # # Validation phase
    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for images, targets in tqdm(val_loader, desc="Validation"):
    #         images = [img.to(DEVICE) for img in images]
    #         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
    #         loss_dict = model(images, targets)
    #         for loss_name, loss_value in loss_dict.items():
    #             logging.info(f"{loss_name}: {loss_value.item()}")
    #         losses = sum(loss for loss in loss_dict.values())
            
    #         val_loss += losses.item()

    # avg_val_loss = val_loss / len(val_loader)
    # print(f"Average Validation Loss: {avg_val_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    print(f"Model saved to {os.path.join(MODEL_SAVE_PATH, MODEL_NAME)}")


logging.info("Training completed.")
