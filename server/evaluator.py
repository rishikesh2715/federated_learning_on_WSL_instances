# server/evaluator.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# === Load test data (once, globally) ===
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# === Evaluate model accuracy ===
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    # print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
