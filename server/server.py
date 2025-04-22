# server/server.py

from flask import Flask, request, jsonify
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.model import MLP
from evaluator import evaluate_model



app = Flask(__name__)
model = MLP()

# Buffer to hold weights from clients
weight_buffer = {}

# Expected clients by port label
EXPECTED_CLIENTS = ["6001", "6002"]
round_counter = 0

# === Helper functions ===
def decode_weights(json_weights):
    return {k: torch.tensor(v) for k, v in json_weights.items()}

def encode_weights(state_dict):
    return {k: v.tolist() for k, v in state_dict.items()}

def average_weights(weight_list):
    avg = {}
    for key in weight_list[0]:
        avg[key] = sum(weights[key] for weights in weight_list) / len(weight_list)
    return avg

# === Flask Routes ===
@app.route("/upload", methods=["POST"])
def upload_model():
    global round_counter

    port = request.args.get("port")
    weights = decode_weights(request.get_json())
    weight_buffer[port] = weights

    print(f"[✓] Received weights from client {port}.")

    # Wait for all clients
    if all(p in weight_buffer for p in EXPECTED_CLIENTS):
        print("[✓] All clients submitted. Averaging weights...")
        averaged = average_weights([weight_buffer[p] for p in EXPECTED_CLIENTS])
        model.load_state_dict(averaged)
        weight_buffer.clear()

        round_counter += 1
        print(f"[✓] Aggregation complete. Evaluating global model (Round {round_counter})...")
        acc = evaluate_model(model)
        print(f"📊 Accuracy after Round {round_counter}: {acc:.2f}%")

        # Log accuracy to file
        with open("server/accuracy_log.txt", "a") as f:
            f.write(f"{round_counter},{acc:.2f}\n")

    return jsonify({"status": "received"})

@app.route("/download", methods=["GET"])
def download_model():
    state_dict = model.state_dict()
    return jsonify(encode_weights(state_dict))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
