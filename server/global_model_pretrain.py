import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client.model import MLP

# === Config ===
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.01
MODEL_SAVE_PATH = "models/global_model_pretrained.pth"
USE_FRACTION = 0.1    # ← use 10% of train set each epoch
os.makedirs("models", exist_ok=True)

# === Data Loader (Full MNIST Train Set) ===
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataset_size = len(train_dataset)
subset_size = int(dataset_size * USE_FRACTION)

# We’ll recreate this sampler each epoch to get a fresh random subset
def make_subset_sampler():
    # torch.randperm gives a random permutation of [0 .. dataset_size-1]
    indices = torch.randperm(dataset_size)[:subset_size].tolist()
    return SubsetRandomSampler(indices)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    # build a DataLoader over a random 10% subset
    sampler = make_subset_sampler()
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    for inputs, labels in loader:
        inputs = inputs.view(inputs.size(0), -1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"🌍 Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

# === Save Pretrained Global Model ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 Pretrained global model saved to {MODEL_SAVE_PATH}")
