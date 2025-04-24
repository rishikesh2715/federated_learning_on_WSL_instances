# server/server.py

from flask import Flask, request, jsonify
import torch, os, sys
from threading import Lock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client.model import MLP
from evaluator import evaluate_model

app = Flask(__name__)
model = MLP()

EXPECTED_CLIENTS = {"6001", "6002"}
current_server_round = 1
server_lock = Lock()

weight_buffer_per_round = {}
ready_clients_per_round = {}

def decode_weights(json_weights):
    return {k: torch.tensor(v) for k, v in json_weights.items()}

def encode_weights(state_dict):
    return {k: v.tolist() for k, v in state_dict.items()}

def average_weights(weight_list):
    keys = weight_list[0].keys()
    return {
        key: torch.stack([w[key] for w in weight_list], dim=0).mean(dim=0)
        for key in keys
    }

@app.route("/upload", methods=["POST"])
def upload_model():
    global current_server_round

    port = request.args.get("port")
    client_round = request.args.get("round", type=int)
    if not port or client_round is None:
        return jsonify({"status": "error", "message": "Missing port or round"}), 400
    if port not in EXPECTED_CLIENTS:
        return jsonify({"status": "error", "message": "Unknown client"}), 400

    with server_lock:
        if client_round != current_server_round:
            return jsonify({"status": "wrong_round", "expected": current_server_round}), 409

        weights = decode_weights(request.get_json())
        weight_buffer_per_round.setdefault(client_round, {})[port] = weights
        print(f"[✓] Received weights from {port} for round {client_round}")

        if len(weight_buffer_per_round[client_round]) == len(EXPECTED_CLIENTS):
            avg_weights = average_weights(list(weight_buffer_per_round[client_round].values()))
            model.load_state_dict(avg_weights)

            acc = evaluate_model(model)
            print(f"📊 Round {client_round} Accuracy: {acc:.2f}%")
            with open("server/accuracy_log.txt", "a") as f:
                f.write(f"{client_round},{acc:.2f}\n")

            weight_buffer_per_round.pop(client_round, None)
            ready_clients_per_round.pop(client_round, None)
            current_server_round += 1
            print(f"[→] Advancing to round {current_server_round}")

    return jsonify({"status": "received"})

@app.route("/ready", methods=["GET"])
def client_ready():
    port = request.args.get("port")
    client_round = request.args.get("round", type=int)
    if not port or client_round is None:
        return jsonify({"go": False}), 400

    with server_lock:
        if client_round != current_server_round:
            return jsonify({"go": False}), 409

        ready_clients_per_round.setdefault(client_round, set()).add(port)
        if ready_clients_per_round[client_round] == EXPECTED_CLIENTS:
            return jsonify({"go": True})
    return jsonify({"go": False})

@app.route("/download", methods=["GET"])
def download_model():
    return jsonify({
        "weights": encode_weights(model.state_dict()),
        "model_round": current_server_round - 1
    })

if __name__ == "__main__":
    os.makedirs(os.path.dirname("server/accuracy_log.txt"), exist_ok=True)
    print(f"🌐 Server ready, starting at round {current_server_round}")
    app.run(host="0.0.0.0", port=5000, debug=True)
