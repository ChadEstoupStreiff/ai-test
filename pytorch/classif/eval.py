import time

from model import ConvNet, ResNet
from torchvision import datasets, transforms

import torch
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FOLDER = "./data/birds/valid"
MODEL_PATH = "./models/birds-121.pth"
MODEL_INPUT_SIZE = 224


def do_evaluation(
    data_folder: str, model_path: str, model_input_size: int, verbose_data: bool = False, class_verbose_data: bool = True
):
    start_time = time.time()
    # Load data
    transform = transforms.Compose(
        [
            transforms.Resize((model_input_size, model_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize with ImageNet standards
        ]
    )
    dataset = datasets.ImageFolder(root=data_folder, transform=transform)
    if verbose_data:
        print(f"Dataset: Number of classes: {len(dataset.classes)}")
        print(f"Dataset: Class to index mapping: \n{dataset.class_to_idx}")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Loading model
    model = ResNet(len(dataset.classes)).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Initialize counters
    class_correct = [0] * len(dataset.classes)
    class_total = [0] * len(dataset.classes)

    # Evaluation loop
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            # Update the counters for each class
            correct = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += correct.item()
                class_total[label] += 1

    # Calculate and print accuracy for each class
    average_accuracy = 0
    for i in range(len(dataset.classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            average_accuracy += accuracy
            print(f"Accuracy of class '{dataset.classes[i]}': {accuracy:.2f}%")
        else:
            print(f"Class '{dataset.classes[i]}' has no samples in the evaluation set.")
    average_accuracy = average_accuracy / (len(dataset.classes))
    print(
        f"Average accuracy: {average_accuracy:.2f}% in {(time.time() - start_time):.3f}s"
    )
    return average_accuracy


if __name__ == "__main__":
    do_evaluation(DATA_FOLDER, MODEL_PATH, MODEL_INPUT_SIZE, verbose_data=True)
