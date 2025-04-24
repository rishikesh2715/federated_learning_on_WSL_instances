import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from client.model import MLP

# === Setup ===
os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load models ===
def load_model(path):
    model = MLP().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

global_model = load_model("models/global_model_final.pth")
client1_model = load_model("models/6001_final.pth")
client2_model = load_model("models/6002_final.pth")

# === Load test data ===
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# === Inference & Metrics ===
def evaluate_and_plot(model, name, in_range=None, save_prefix=""):
    all_preds = []
    all_labels = []
    images_to_show = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.view(images.size(0), -1))
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Pick a few correct samples (both in-range and out-of-range)
            if len(images_to_show) < 25:
                correct_mask = preds == labels
                correct_imgs = images[correct_mask]
                correct_labels = labels[correct_mask]
                correct_preds = preds[correct_mask]
                for i in range(len(correct_imgs)):
                    if len(images_to_show) >= 25:
                        break
                    if in_range is None or correct_labels[i].item() in in_range or correct_preds[i].item() in in_range:
                        images_to_show.append((correct_imgs[i].cpu(), correct_labels[i].item(), correct_preds[i].item()))

    # === Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"{name} — Confusion Matrix")
    plt.savefig(f"results/{save_prefix}_confusion_matrix.png")
    plt.close()

    # === Sample Grid ===
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for ax, (img, true_label, pred_label) in zip(axes.flatten(), images_to_show):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"T:{true_label} / P:{pred_label}", fontsize=10, color="green" if true_label == pred_label else "red")
        ax.axis("off")
    plt.suptitle(f"{name} — Sample Predictions")
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}_sample_predictions.png")
    plt.close()

# === Run Inference for All Models ===
evaluate_and_plot(global_model, "Global Model", in_range=None, save_prefix="global")
evaluate_and_plot(client1_model, "Client 6001", in_range=list(range(0, 5)), save_prefix="client1")
evaluate_and_plot(client2_model, "Client 6002", in_range=list(range(5, 10)), save_prefix="client2")
