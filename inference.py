import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from client.model import MLP

# === Setup ===
os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Font Setup ===
plt.rcParams.update({
    'font.family': 'Courier New',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})

# === Load models ===
def load_model(path):
    model = MLP().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

global_model  = load_model("models/global_model_pre.pth")
client1_model = load_model("models/client1_FedProx_6001_Final.pth")
client2_model = load_model("models/client2_FedProx_6002_Final.pth")

# === Load test data ===
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# === Inference & Plot ===
def evaluate_and_plot(model, name, save_prefix=""):
    # run inference (same as before)…
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.view(images.size(0), -1))
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # compute CM
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))

    # plot with default Blues cmap
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="PuBu", values_format="d")
    n = cm.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    ax.grid(
        which = 'minor',
        color = 'gray',
        linestyle = '-',
        linewidth = 1,
        alpha = 0.4
    )

    ax.tick_params(which='minor', length=0)

    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_aspect('equal')

    # save as a tight square
    out_path = f"results/{save_prefix}_confusion_matrix.pdf"
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close(fig)


# === Run for all models ===
evaluate_and_plot(global_model,  "Global Model",  save_prefix="global")
evaluate_and_plot(client1_model, "Client 6001",   save_prefix="client1")
evaluate_and_plot(client2_model, "Client 6002",   save_prefix="client2")
