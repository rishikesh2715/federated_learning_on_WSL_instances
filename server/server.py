# server/server.py

from flask import Flask, request, jsonify
import torch, os, sys
from threading import Lock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client.model import MLP
from evaluator import evaluate_model

from client.model import save_model

app = Flask(__name__)
model = MLP()
model.load_state_dict(torch.load("models/global_model_pretrained.pth"))


NUM_ROUNDS = 20


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

            loss, acc = evaluate_model(model)
            print(f"📊 Round {client_round} - Global Acc: {acc:.2f}% | Loss: {loss:.4f}")

            # Save accuracy to a log file
            log_path = "server/global_log_pre.csv"
            write_header = not os.path.exists(log_path)

            with open(log_path, "a") as f:
                if write_header:
                    f.write("round,loss,accuracy\n")
                f.write(f"{client_round},{loss:.4f},{acc:.2f}\n")

            # ✅ Only save after final round
            if client_round == NUM_ROUNDS:
                os.makedirs("models", exist_ok=True)
                save_model(model, "models/global_model_pre.pth")
                print(f"💾 Saved global model after final round {client_round}.")




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
