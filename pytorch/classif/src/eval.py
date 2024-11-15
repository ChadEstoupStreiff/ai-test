import time
from typing import Any

import torch
from model import ResNet
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FOLDER = "./resnet_data/eval"
MODEL_PATH = "./models/resnet_v1-4.pth"
MODEL_INPUT_SIZE = (128, 646)
BINARY_MODEL_POSITIVE_INDEX = 0


def do_evaluation(model: Any, dataset: Any, binary_prblm_positive_index: int = 1):
    start_time = time.time()

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Initialize counters
    class_correct = [0] * len(dataset.classes)
    class_total = [0] * len(dataset.classes)
    all_preds = []
    all_labels = []

    # Evaluation loop
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update the counters for each class
            correct = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += correct.item()
                class_total[label] += 1

    # Calculate accuracy for each class
    accuracies = {}
    for i in range(len(dataset.classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            accuracies[dataset.classes[i]] = [
                accuracy,
                class_correct[i],
                class_total[i],
            ]
        else:
            print(f"Class '{dataset.classes[i]}' has no samples in the evaluation set.")

    weighted_accuracy = sum(class_correct) / sum(class_total) * 100
    accuracy = sum(
        [
            100 * correct / total if total > 0 else 0
            for correct, total in zip(class_correct, class_total)
        ]
    ) / len(dataset.classes)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(dataset.classes))), pos_label=binary_prblm_positive_index
    )
    _, _, weighted_f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", pos_label=binary_prblm_positive_index
    )

    precision = [e * 100 for e in precision]
    recall = [e * 100 for e in recall]
    f1_score = [e * 100 for e in f1_score]
    weighted_f1_score = weighted_f1_score * 100

    print(f"Accuracies: {accuracies}")
    print(f"Accuracy: {accuracy:.3f}%")
    print(f"Weighted accuracy: {weighted_accuracy:.3f}%")
    print(f"Precisions: {precision}")
    print(f"Precision: {sum(precision)/len(precision)}")
    print(f"Recalls: {recall}")
    print(f"Recall: {sum(recall)/len(recall)}")
    print(f"F1-scores: {f1_score}")
    print(f"F1-score: {sum(f1_score)/len(f1_score)}")
    print(f"Weighted F1-score: {weighted_f1_score:.3f}")
    print(f"Evaluation done in {(time.time() - start_time):.3f}s")

    return {
        "accuracies": accuracies,
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "precisions": precision,
        "precision": sum(precision) / len(precision),
        "recalls": recall,
        "recall": sum(recall) / len(recall),
        "f1_scores": f1_score,
        "f1_score": sum(f1_score) / len(f1_score),
        "weighted_f1_score": weighted_f1_score,
    }


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize with ImageNet standards
        ]
    )
    dataset = datasets.ImageFolder(root=DATA_FOLDER, transform=transform)
    print(f"Dataset: Number of classes: {len(dataset.classes)}")
    print(f"Dataset: Class to index mapping: \n{dataset.class_to_idx}")

    model = ResNet(MODEL_INPUT_SIZE, len(dataset.classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    print(do_evaluation(model, dataset, binary_prblm_positive_index=BINARY_MODEL_POSITIVE_INDEX))
