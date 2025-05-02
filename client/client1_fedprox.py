# client/client1.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import requests
import time
from torch.utils.data import random_split
from model import MLP
from model import save_model
import os

# === Config ===
DIGIT_RANGE = list(range(0, 5))
PORT = "6001"
SERVER_URL = "http://172.19.32.1:5000"
EPOCHS_PER_ROUND = 1
NUM_ROUNDS = 20
BATCH_SIZE = 64
LR = 0.01
MU = 1  # FedProx regularization strength


# === Load Data ===
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
filtered_data = [(x, y) for x, y in full_dataset if y in DIGIT_RANGE]


# Split into train and val (80/20)
train_size = int(0.8 * len(filtered_data))
val_size = len(filtered_data) - train_size
train_data, val_data = random_split(filtered_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)


# === Serialize / Deserialize ===
def serialize_weights(state_dict):
    return {k: v.tolist() for k, v in state_dict.items()}

def deserialize_weights(json_weights):
    return {k: torch.tensor(v) for k, v in json_weights.items()}

def train_and_validate(model, optimizer, criterion, train_loader, val_loader, global_weights=None):
    # Train
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # === FedProx penalty ===
        if global_weights is not None:
            prox_reg = 0.0
            for name, param in model.named_parameters():
                prox_reg += ((param - global_weights[name].to(param.device)) ** 2).sum()
            loss += (MU / 2) * prox_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = 100.0 * correct / total

    # === Validation (unchanged) ===
    model.eval()
    with torch.no_grad():
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in val_loader:
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = total_loss / total
    val_acc = 100.0 * correct / total

    return train_loss, train_acc, val_loss, val_acc




# === Round Sync ===
def wait_for_turn(port, round_num):
    while True:
        try:
            r = requests.get(f"{SERVER_URL}/ready", params={"port": port, "round": round_num}, timeout=5)
            if r.ok and r.json().get("go"):
                return
            print(f"[{port}] ⏳ Waiting for round {round_num}...")
        except Exception as e:
            print(f"[{port}] ⚠️ Sync error: {e}")
        time.sleep(2)

# === Federated Learning Loop ===
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n🔁 [{PORT}] Round {round_num}/{NUM_ROUNDS}")

    # Step 0: Sync
    wait_for_turn(PORT, round_num)

    # Step 1: Download global model
    try:
        res = requests.get(f"{SERVER_URL}/download", timeout=5)
        model.load_state_dict(deserialize_weights(res.json()["weights"]))
        print(f"[{PORT}] ✅ Downloaded global model.")
    except Exception as e:
        print(f"[{PORT}] ❌ Download failed: {e}")
        break

    # Step 2: Local Training
    try:
        print(f"[{PORT}] ⏳ Starting local training for round {round_num}...")
        # Save a frozen copy of global weights for FedProx
        global_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

        # Pass global_weights to training
        train_loss, train_acc, val_loss, val_acc = train_and_validate(
            model, optimizer, criterion, train_loader, val_loader, global_weights
)
        print(f"[{PORT}] ✅ Round {round_num} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Save to CSV
        log_path = f"client/pre_{PORT}_log.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if write_header:
                f.write("round,train_loss,train_acc,val_loss,val_acc\n")
            f.write(f"{round_num},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}\n")


    except Exception as e:
        print(f"[{PORT}] ❌ Failed during local training for round {round_num}: {e}")
        break


    # Step 3: Upload
    try:
        payload = serialize_weights(model.state_dict())
        res = requests.post(f"{SERVER_URL}/upload", params={"port": PORT, "round": round_num}, json=payload, timeout=10)
        if res.ok:
            print(f"[{PORT}] ✅ Uploaded model.")
        else:
            print(f"[{PORT}] ❌ Upload failed (status {res.status_code}): {res.text}")
            break
    except Exception as e:
        print(f"[{PORT}] ❌ Upload exception: {e}")
        break

print(f"🏁 [{PORT}] Finished all rounds.")

os.makedirs("models", exist_ok=True)
save_model(model, f"models/client1_FedProx_{PORT}_final.pth")
print(f"[{PORT}] 💾 Saved final model.")
