# client/client1.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import requests
import time
from model import MLP

# === Config ===
DIGIT_RANGE = list(range(0, 5))
PORT = "6001"
SERVER_URL = "http://172.19.32.1:5000"  # UPDATE this IP if needed
EPOCHS_PER_ROUND = 1
NUM_ROUNDS = 20
BATCH_SIZE = 64
LR = 0.01

# === Load Data ===
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
filtered_data = [(x, y) for x, y in full_dataset if y in DIGIT_RANGE]
train_loader = torch.utils.data.DataLoader(filtered_data, batch_size=BATCH_SIZE, shuffle=True)

# === Serialize / Deserialize ===
def serialize_weights(state_dict):
    return {k: v.tolist() for k, v in state_dict.items()}

def deserialize_weights(json_weights):
    return {k: torch.tensor(v) for k, v in json_weights.items()}

# === Local Training Loop ===
def train_local(model, optimizer, criterion):
    model.train()
    for _ in range(EPOCHS_PER_ROUND):
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def wait_for_turn(port):
    while True:
        try:
            r = requests.get(f"{SERVER_URL}/ready?port={port}")
            if r.json().get("go"):
                break
            print(f"[{port}] ⏳ Waiting for other client...")
        except Exception as e:
            print(f"[{port}] ❌ Sync error: {e}")
        time.sleep(1)

# === Federated Learning Rounds ===
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

for round in range(NUM_ROUNDS):
    print(f"\n🔁 Round {round + 1}/{NUM_ROUNDS}")

    # === STEP 0: Synchronize with the other client ===
    wait_for_turn(PORT)

    # Step 1: Download latest global model
    try:
        response = requests.get(f"{SERVER_URL}/download")
        global_weights = deserialize_weights(response.json())
        model.load_state_dict(global_weights)
        print(f"[{PORT}] ✅ Downloaded global model.")
    except Exception as e:
        print(f"[{PORT}] ❌ Failed to download global model: {e}")
        break

    # Step 2: Train locally
    train_local(model, optimizer, criterion)
    print(f"[{PORT}] ✅ Local training done.")

    # Step 3: Upload new weights
    try:
        payload = serialize_weights(model.state_dict())
        response = requests.post(f"{SERVER_URL}/upload?port={PORT}", json=payload)
        print(f"[{PORT}] ✅ Uploaded model to server.")
    except Exception as e:
        print(f"[{PORT}] ❌ Failed to upload model: {e}")
        break

    # Short sleep to simulate network delay and stagger clients
    time.sleep(1)
