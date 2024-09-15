import os
import time

import torchvision
from eval import do_evaluation
from model import ConvNet, ResNet
from torchvision import datasets, transforms
from gpubalancer import GPUBalancerClient
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

# Device configuration
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################ Hyper-parameters
SAVE_PATH = "../models"
DATA_PATH = "../data/birds"
MODEL_NAME = "birds3"

MODEL_INPUT_SIZE = 224

NUM_EPOCHS = 3000
BATCH_SIZE = 256
LEARNING_RATE = 0.00005

VERBOSE_PERIOD = 1
EVAL_PERIOD = 1

################ Sport data loading
writer = SummaryWriter(f"../runs/{MODEL_NAME}")
logging.info("Loading dataset...")
training_transform = transforms.Compose(
    [
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet standards
    ]
)
transform = transforms.Compose(
    [
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet standards
    ]
)

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_PATH, "train"), transform=training_transform
)
eval_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_PATH, "eval"), transform=transform
)
test_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_PATH, "test"), transform=transform
)
logging.info(f"Train dataset: Number of classes: {len(train_dataset.classes)}")
logging.info(f"Train dataset: Class to index mapping: \n{train_dataset.class_to_idx}")
logging.info(f"Eval dataset: Number of classes: {len(train_dataset.classes)}")
logging.info(f"Eval dataset: Class to index mapping: \n{train_dataset.class_to_idx}")
logging.info(f"Test dataset: Number of classes: {len(test_dataset.classes)}")
logging.info(f"Test dataset: Class to index mapping: \n{test_dataset.class_to_idx}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

examples = iter(test_loader)
example_data, example_targets = next(examples)
classes = train_dataset.classes

img_grid = torchvision.utils.make_grid(example_data)
writer.add_image("mnist_images", img_grid)

################ Custom Model
with GPUBalancerClient(importance=-1,
    max_gpu_load=11000,
    title="PyTorch test",
    description="C'est Chad qui fou la merde :)",
    host="10.10.41.2", port=8043):

    logging.info("Creating model...")
    model = ResNet(len(train_dataset.classes)).to(device)

    ################ Loss & Optimizer

    lossfct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    writer.add_graph(model, example_data.to(device))
    writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], 0)

    ################ Training
    logging.info("Started training...")
    for epoch in range(NUM_EPOCHS):

        start_time = time.time()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1:3}/{NUM_EPOCHS}"):
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
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch+1)

        # Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / total_samples
        epoch_accuracy = running_correct / total_samples

        if epoch % VERBOSE_PERIOD == 0:
            logging.info(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f} in {(time.time() - start_time):.3f}s"
            )

        # Log metrics
        writer.add_scalar("train_loss", epoch_loss, epoch)
        writer.add_scalar("train_accuracy", epoch_accuracy, epoch)

        if epoch % EVAL_PERIOD == 0:
            logging.info("Evaluating model...")

            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            PATH = os.path.join(SAVE_PATH, f"{MODEL_NAME}-{epoch+1}.pth")
            torch.save(model.state_dict(), PATH)
            model.eval()
            eval_accuracy = do_evaluation(
                model, eval_dataset
            )
            model.train()
            writer.add_scalar("eval_accuracy", eval_accuracy, epoch)

    logging.info("Finished Training. Saving model...")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    PATH = os.path.join(SAVE_PATH, f"{MODEL_NAME}-final.pth")
    torch.save(model.state_dict(), PATH)
