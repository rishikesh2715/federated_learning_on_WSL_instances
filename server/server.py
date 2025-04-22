# server/server.py

from flask import Flask, request, jsonify
import torch
from client.model import MLP

app = Flask(__name__)
model = MLP()

# Stores incoming weights from clients
weight_buffer = {}

# Identify clients by their ports
EXPECTED_CLIENTS = ["6001", "6002"]

# === Helper Functions ===
def decode_weights(json_weights):
    return {k: torch.tensor(v) for k, v in json_weights.items()}

def average_weights(weight_list):
    avg = {}
    for key in weight_list[0]:
        avg[key] = sum(weights[key] for weights in weight_list) / len(weight_list)
    return avg

# === Routes ===

@app.route("/upload", methods=["POST"])
def upload_model():
    port = request.args.get("port")
    weights = decode_weights(request.get_json())

    weight_buffer[port] = weights
    print(f"[✓] Received weights from client {port}")

    # Wait for all clients
    if all(p in weight_buffer for p in EXPECTED_CLIENTS):
        print("[✓] All clients submitted. Averaging weights...")
        averaged = average_weights([weight_buffer[p] for p in EXPECTED_CLIENTS])
        model.load_state_dict(averaged)
        weight_buffer.clear()
        print("[✓] Averaging complete. Global model updated.")

    return jsonify({"status": "received"})

@app.route("/download", methods=["GET"])
def download_model():
    state_dict = model.state_dict()
    serialized = {k: v.tolist() for k, v in state_dict.items()}
    print("[↓] Sent global model to client.")
    return jsonify(serialized)


if __name__ == "__main__":
    app.run(port=5000)
