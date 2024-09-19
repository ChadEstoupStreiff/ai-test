import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from eval import do_evaluation
from model import ResNet
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################ Hyper-parameters
SAVE_PATH = "./models/"
DATA_PATH = "./resnet_data/"
MODEL_NAME = "resnet_v3"
BINARY_MODEL_POSITIVE_INDEX = 0

START_EPOCH = 1
MODEL_CHECKPOINT = f"models/resnet_v2_{START_EPOCH}.pth"  # None to start from scratch model
MODEL_CHECKPOINT = None  # None to start from scratch model

MODEL_INPUT_SIZE = (128, 256)
# MODEL_INPUT_SIZE = (128, 646)

NUM_EPOCHS = 3000
BATCH_SIZE = 512
LEARNING_RATE = 0.00005

VERBOSE_PERIOD = 1
EVAL_PERIOD = 1
# None for input in shell, path for a file
# LOG_FILE = None
LOG_FILE = "train.log"  

if LOG_FILE is not None:
    log_file = open(LOG_FILE, "a")
    sys.stdout = log_file
    sys.stderr = log_file

writer = SummaryWriter(f"./runs/{MODEL_NAME}")


################ data loading

print("Loading dataset...")
training_transform = transforms.Compose(
    [
        # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.Resize((MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])),
        # transforms.ColorJitter(
        #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        # ),  # Optional for spectrogram
        # # transforms.RandomRotation(10),  # Small rotations for invariance
        # transforms.RandomPerspective(
        #     distortion_scale=0.2, p=0.5
        # ),  # Simulates perspective changestransforms.ToTensor(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet stats
    ]
)
transform = transforms.Compose(
    [
        transforms.Resize((MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet standards
    ]
)

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_PATH, "train"), transform=training_transform
)
# To counter imbalance dataset
class_counts = [
    len(np.where(np.array(train_dataset.targets) == t)[0])
    for t in np.unique(train_dataset.targets)
]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
print(class_weights)
sample_weights = class_weights[train_dataset.targets]
print(sample_weights)
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

eval_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_PATH, "eval"), transform=transform
)

print(f"Train dataset: Number of classes: {len(train_dataset.classes)}")
print(f"Train dataset: Class to index mapping: \n{train_dataset.class_to_idx}")
print(f"Eval dataset: Number of classes: {len(train_dataset.classes)}")
print(f"Eval dataset: Class to index mapping: \n{train_dataset.class_to_idx}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=sampler
)

examples = iter(train_loader)
example_data, example_targets = next(examples)
classes = train_dataset.classes

img_grid = torchvision.utils.make_grid(example_data)
writer.add_image("images", img_grid)

################ Custom Model
print(f"Creating model {MODEL_NAME} ...")
model = ResNet(MODEL_INPUT_SIZE, len(train_dataset.classes)).to(device)
if MODEL_CHECKPOINT is not None:
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, weights_only=False))
    print(f"Checkpoint loaded: {MODEL_CHECKPOINT}")
model.eval()
summary(model, (3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]))

################ Loss & Optimizer

lossfct = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
writer.add_graph(model, example_data.to(device))
writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], 0)

################ Training
print("Started training...")
model.train()
for epoch in range(START_EPOCH, NUM_EPOCHS):
    start_time = time.time()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    for i, (images, labels) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch+1:3}/{NUM_EPOCHS}",
        mininterval=60,
    ):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = lossfct(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    scheduler.step()
    writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch + 1)

    # Compute average loss and accuracy for the epoch
    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    if epoch % VERBOSE_PERIOD == 0:
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f} in {(time.time() - start_time):.3f}s"
        )

    # Log metrics
    writer.add_scalar("train_loss", epoch_loss, epoch)
    writer.add_scalar("train_accuracy", epoch_accuracy, epoch)

    if epoch % EVAL_PERIOD == 0:
        print("Evaluating model...")

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        PATH = os.path.join(SAVE_PATH, f"{MODEL_NAME}-{epoch+1}.pth")
        torch.save(model.state_dict(), PATH)
        model.eval()
        metrics = do_evaluation(model, eval_dataset, binary_prblm_positive_index=BINARY_MODEL_POSITIVE_INDEX)
        model.train()
        writer.add_scalar("eval_accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("eval_weighted_accuracy", metrics["weighted_accuracy"], epoch)
        writer.add_scalar("eval_precision", metrics["precision"], epoch)
        writer.add_scalar("eval_recall", metrics["recall"], epoch)
        writer.add_scalar("eval_f1_score", metrics["f1_score"], epoch)
        writer.add_scalar("eval_weighted_f1_score", metrics["weighted_f1_score"], epoch)

print("Finished Training. Saving model...")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
PATH = os.path.join(SAVE_PATH, f"{MODEL_NAME}-final.pth")
torch.save(model.state_dict(), PATH)
