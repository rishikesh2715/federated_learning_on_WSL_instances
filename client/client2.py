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


def wait_for_turn(port, target_round):
    """Polls the server until it's ready for the target_round."""
    while True:
        try:
            # Pass the round the client is ready for
            params = {'port': port, 'round': target_round}
            r = requests.get(f"{SERVER_URL}/ready", params=params, timeout=10) # Add timeout
            response_json = r.json()

            if r.status_code == 200 and response_json.get("go"):
                print(f"[{port}] ✅ Server ready for round {target_round}.")
                break
            elif r.status_code == 409: # Conflict (likely wrong round)
                 server_expects = response_json.get("message", "unknown state")
                 print(f"[{port}] ⏳ Waiting: Server not ready for round {target_round}. {server_expects}. Retrying...")
            else:
                 # Handle other potential non-200 responses if needed
                 print(f"[{port}] ⏳ Waiting for server for round {target_round}... (Status: {r.status_code}, Response: {response_json})")

        except requests.exceptions.Timeout:
             print(f"[{port}] ⏳ Timeout connecting to server for /ready check. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"[{port}] ❌ Sync error (round {target_round}): {e}. Retrying...")
        except Exception as e:
             print(f"[{port}] ❌ Unexpected error in wait_for_turn (round {target_round}): {e}")

        time.sleep(2) # Increase sleep slightly to reduce spamming server


# === Federated Learning Rounds ===
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# NUM_ROUNDS defined in config
for i in range(NUM_ROUNDS):
    current_round = i + 1 # Rounds are 1-based

    print(f"\n--- [{PORT}] Attempting Round {current_round}/{NUM_ROUNDS} ---")

    # === STEP 0: Synchronize: Wait for server to be ready for THIS round ===
    wait_for_turn(PORT, current_round)

    # === Step 1: Download latest global model ===
    try:
        response = requests.get(f"{SERVER_URL}/download", timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        global_weights = deserialize_weights(data["weights"])
        model_round = data.get("model_round", "N/A") # Get the round the model is from
        model.load_state_dict(global_weights)
        print(f"[{PORT}] ✅ Downloaded global model (from round {model_round}).")
    except Exception as e:
        print(f"[{PORT}] ❌ Failed to download global model for round {current_round}: {e}")
        print("Stopping client.")
        break # Stop client if download fails

    # === Step 2: Train locally ===
    try:
        print(f"[{PORT}] ⏳ Starting local training for round {current_round}...")
        train_local(model, optimizer, criterion)
        print(f"[{PORT}] ✅ Local training done for round {current_round}.")
    except Exception as e:
        print(f"[{PORT}] ❌ Failed during local training for round {current_round}: {e}")
        print("Stopping client.")
        break # Stop client if training fails


    # === Step 3: Upload new weights ===
    retries = 3
    upload_success = False
    for attempt in range(retries):
         try:
            payload = serialize_weights(model.state_dict())
            # Pass the current round number in the upload request
            params = {'port': PORT, 'round': current_round}
            response = requests.post(f"{SERVER_URL}/upload", params=params, json=payload, timeout=30) # Longer timeout for upload

            if response.status_code == 200:
                print(f"[{PORT}] ✅ Uploaded model to server for round {current_round}.")
                upload_success = True
                break # Success
            elif response.status_code == 409: # Conflict, likely server advanced or client is lagging
                 server_expects = response.json().get("message", "unknown state")
                 print(f"[{PORT}] ⚠️ Upload rejected for round {current_round}. {server_expects}. (Attempt {attempt+1}/{retries})")
                 # If server is ahead, maybe we need to skip or reset? For now, just retry/fail.
                 time.sleep(3) # Wait before retrying
            else:
                response.raise_for_status() # Raise exception for other errors

         except requests.exceptions.Timeout:
             print(f"[{PORT}] ❌ Timeout uploading model for round {current_round}. (Attempt {attempt+1}/{retries})")
             time.sleep(3)
         except requests.exceptions.RequestException as e:
             print(f"[{PORT}] ❌ Failed to upload model for round {current_round}: {e}. (Attempt {attempt+1}/{retries})")
             time.sleep(3) # Wait before retrying
         except Exception as e:
             print(f"[{PORT}] ❌ Unexpected error during upload (round {current_round}): {e}")
             break # Don't retry on unexpected errors

    if not upload_success:
        print(f"[{PORT}] ❌ Failed to upload model for round {current_round} after {retries} attempts. Stopping client.")
        break

    # Short sleep potentially helps stagger clients slightly if needed, but strict sync is primary goal
    # time.sleep(1)

print(f"--- [{PORT}] Client finished ---")
