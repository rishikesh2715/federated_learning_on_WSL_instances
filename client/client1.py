# client/client1.py (updated with global model sync)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import requests
from model import MLP

# === Config ===
DIGIT_RANGE = list(range(0, 5))
SERVER_URL = "http://localhost:5000"
PORT = "6001"
EPOCHS = 3
BATCH_SIZE = 64
LR = 0.01

# === Load & Filter Data ===
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
filtered_data = [(x, y) for x, y in full_dataset if y in DIGIT_RANGE]
train_loader = torch.utils.data.DataLoader(filtered_data, batch_size=BATCH_SIZE, shuffle=True)

# === Initialize Model ===
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# === Train ===
for epoch in range(EPOCHS):
    for inputs, labels in train_loader:
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"[{PORT}] Epoch {epoch+1} done")

# === Send Local Weights ===
def serialize_weights(state_dict):
    return {k: v.tolist() for k, v in state_dict.items()}

response = requests.post(f"{SERVER_URL}/upload?port={PORT}", json=serialize_weights(model.state_dict()))
print(f"[{PORT}] Sent weights to server.")

# === Download Global Weights ===
def deserialize_weights(json_weights):
    return {k: torch.tensor(v) for k, v in json_weights.items()}

response = requests.get(f"{SERVER_URL}/download")
global_weights = deserialize_weights(response.json())
model.load_state_dict(global_weights)
print(f"[{PORT}] Synced with global model.")
