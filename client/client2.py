# client/client2.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import requests
import time
from model import MLP

# === Config ===
DIGIT_RANGE = list(range(5, 10))
PORT = "6002"
SERVER_URL = "http://172.19.32.1:5000"
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

# === Local Training ===
def train_local(model, optimizer, criterion, dataloader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs = inputs.view(inputs.size(0), -1)
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
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


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
        loss, acc = train_local(model, optimizer, criterion, train_loader)
        print(f"[{PORT}] ✅ Local training done. Loss: {loss:.4f}, Acc: {acc:.2f}%")

        # Save to CSV log
        with open(f"client/{PORT}_log.csv", "a") as f:
            f.write(f"{round_num},{loss:.4f},{acc:.2f}\n")

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
