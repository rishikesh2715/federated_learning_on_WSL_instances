import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client.model import MLP

# === Config ===
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.01
MODEL_SAVE_PATH = "models/global_model_pretrained.pth"
os.makedirs("models", exist_ok=True)

# === Data Loader (Full MNIST Train Set) ===
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
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
