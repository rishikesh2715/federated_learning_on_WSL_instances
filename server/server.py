# # server/server.py

# from flask import Flask, request, jsonify
# import torch
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from client.model import MLP
# from evaluator import evaluate_model



# app = Flask(__name__)
# model = MLP()

# # Buffer to hold weights from clients
# weight_buffer = {}

# # Expected clients by port label
# EXPECTED_CLIENTS = ["6001", "6002"]
# round_counter = 0

# # === Helper functions ===
# def decode_weights(json_weights):
#     return {k: torch.tensor(v) for k, v in json_weights.items()}

# def encode_weights(state_dict):
#     return {k: v.tolist() for k, v in state_dict.items()}

# def average_weights(weight_list):
#     avg = {}
#     for key in weight_list[0]:
#         avg[key] = sum(weights[key] for weights in weight_list) / len(weight_list)
#     return avg

# # === Flask Routes ===
# @app.route("/upload", methods=["POST"])
# def upload_model():
#     global round_counter

#     port = request.args.get("port")
#     weights = decode_weights(request.get_json())
#     weight_buffer[port] = weights

#     print(f"[✓] Received weights from client {port}.")

#     # Wait for all clients
#     if all(p in weight_buffer for p in EXPECTED_CLIENTS):
#         print("[✓] All clients submitted. Averaging weights...")
#         averaged = average_weights([weight_buffer[p] for p in EXPECTED_CLIENTS])
#         model.load_state_dict(averaged)
#         weight_buffer.clear()

#         round_counter += 1
#         print(f"[✓] Aggregation complete. Evaluating global model (Round {round_counter})...")
#         acc = evaluate_model(model)
#         print(f"📊 Accuracy after Round {round_counter}: {acc:.2f}%")

#         # Log accuracy to file
#         with open("server/accuracy_log.txt", "a") as f:
#             f.write(f"{round_counter},{acc:.2f}\n")

#     return jsonify({"status": "received"})


# ready_clients = set()

# @app.route("/ready", methods=["GET"])
# def client_ready():
#     port = request.args.get("port")
#     ready_clients.add(port)

#     if ready_clients == set(EXPECTED_CLIENTS):
#         ready_clients.clear()
#         return jsonify({"go": True})
#     return jsonify({"go": False})



# @app.route("/download", methods=["GET"])
# def download_model():
#     state_dict = model.state_dict()
#     return jsonify(encode_weights(state_dict))

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)



# server/server.py

from flask import Flask, request, jsonify
import torch
import sys, os
from threading import Lock
# (Keep existing sys.path modifications)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.model import MLP # Assuming model.py is in client directory relative to project root
from evaluator import evaluate_model # Assuming evaluator.py is in the same directory as server.py or accessible

app = Flask(__name__)
model = MLP()

# --- State Variables ---
# Use dictionaries keyed by round number for state
weight_buffer_per_round = {} # { round_num: {port: weights} }
ready_clients_per_round = {} # { round_num: set(ports) }
EXPECTED_CLIENTS = {"6001", "6002"} # Use a set for easier comparison
current_server_round = 1 # Start expecting round 1
server_lock = Lock() # To prevent race conditions when updating shared state

# === Helper functions === (Keep decode_weights, encode_weights, average_weights as before)
def decode_weights(json_weights):
    return {k: torch.tensor(v) for k, v in json_weights.items()}

def encode_weights(state_dict):
    return {k: v.tolist() for k, v in state_dict.items()}

def average_weights(weight_list):
    if not weight_list: return {}
    avg = {}; num_clients = len(weight_list)
    keys = weight_list[0].keys()
    for key in keys:
        tensors = [weights[key] for weights in weight_list if key in weights and isinstance(weights[key], torch.Tensor)]
        if len(tensors) == num_clients: # Ensure all clients provided the key
            avg[key] = torch.stack(tensors, dim=0).mean(dim=0) # Use mean for simplicity
        else: print(f"Warning: Key '{key}' missing or invalid in some weights for averaging.")
    return avg


# === Flask Routes ===

@app.route("/upload", methods=["POST"])
def upload_model():
    global current_server_round # Allow modification

    port = request.args.get("port")
    client_round = request.args.get("round", type=int)

    if not port or client_round is None:
        return jsonify({"status": "error", "message": "Missing client port or round number"}), 400

    if port not in EXPECTED_CLIENTS:
         return jsonify({"status": "error", "message": f"Unrecognized client port: {port}"}), 400

    # --- Critical Section Start ---
    with server_lock:
        # Check if the upload is for the round the server is currently expecting
        if client_round != current_server_round:
            print(f"[!] Client {port} tried to upload for round {client_round}, but server expects round {current_server_round}.")
            # Determine if client is ahead or behind
            status = "too_early" if client_round > current_server_round else "too_late"
            return jsonify({"status": status, "message": f"Server expects round {current_server_round}"}), 409 # Conflict

        try:
            weights = decode_weights(request.get_json())
        except Exception as e:
            print(f"[✗] Error decoding weights from {port} for round {client_round}: {e}")
            return jsonify({"status": "error", "message": "Failed to decode weights"}), 400

        # Initialize round buffer if it doesn't exist
        if client_round not in weight_buffer_per_round:
            weight_buffer_per_round[client_round] = {}

        # Store weights
        weight_buffer_per_round[client_round][port] = weights
        print(f"[✓] Received weights from client {port} for round {client_round}.")

        # Check if all clients have submitted for the current round
        if len(weight_buffer_per_round[client_round]) == len(EXPECTED_CLIENTS):
            print(f"[✓] All clients submitted for round {client_round}. Averaging weights...")
            try:
                weights_to_avg = list(weight_buffer_per_round[client_round].values())
                averaged = average_weights(weights_to_avg)

                if averaged: # Check if averaging was successful
                    model.load_state_dict(averaged)
                    print(f"[✓] Aggregation complete for round {client_round}.")

                    # Evaluate Model
                    try:
                         acc = evaluate_model(model)
                         print(f"📊 Accuracy after Round {client_round}: {acc:.2f}%")
                         # Log accuracy
                         log_dir = os.path.dirname(__file__)
                         log_path = os.path.join(log_dir, "accuracy_log.txt")
                         os.makedirs(os.path.dirname(log_path), exist_ok=True)
                         with open(log_path, "a") as f:
                               f.write(f"{client_round},{acc:.2f}\n")
                    except Exception as e:
                         print(f"[✗] Error during model evaluation or logging: {e}")

                    # --- Clean up and Advance Round ---
                    # Clean up state for the completed round
                    del weight_buffer_per_round[client_round]
                    if client_round in ready_clients_per_round:
                         del ready_clients_per_round[client_round]

                    # Advance the server to the next round
                    current_server_round += 1
                    print(f"[→] Server advanced to expect round {current_server_round}.")

                else:
                     print(f"[✗] Averaging failed for round {client_round}. Check client weight structure.")
                     # Potentially need error handling strategy here - maybe clear weights and ask clients to retry?

            except Exception as e:
                print(f"[✗] Error during aggregation or model update for round {client_round}: {e}")

    # --- Critical Section End ---
    return jsonify({"status": "received"})


@app.route("/ready", methods=["GET"])
def client_ready():
    port = request.args.get("port")
    client_round = request.args.get("round", type=int) # Client declares which round it's ready for

    if not port or client_round is None:
        return jsonify({"go": False, "message": "Missing client port or round number"}), 400

    if port not in EXPECTED_CLIENTS:
        return jsonify({"go": False, "message": f"Unrecognized client port: {port}"}), 400

    # --- Critical Section Start ---
    with server_lock:
        # Check if client is ready for the round the server expects
        if client_round != current_server_round:
            print(f"[!] Client {port} reported ready for round {client_round}, but server expects {current_server_round}.")
            # Let client know if it needs to wait or if it missed a round (though upload logic handles missed rounds better)
            return jsonify({"go": False, "message": f"Server expects round {current_server_round}"}), 409 # Conflict

        # Initialize round readiness set if it doesn't exist
        if client_round not in ready_clients_per_round:
            ready_clients_per_round[client_round] = set()

        # Add client to the set for the declared round
        ready_clients_per_round[client_round].add(port)
        print(f"[?] Client {port} reported ready for round {client_round}. Current ready set for round {client_round}: {ready_clients_per_round[client_round]}")

        # Check if all expected clients are ready *for this specific round*
        go_signal = False
        if ready_clients_per_round[client_round] == EXPECTED_CLIENTS:
            print(f"[✓] All expected clients are ready for round {client_round}. Signaling 'go'.")
            go_signal = True
            # DO NOT clear readiness here; clear after upload/aggregation

    # --- Critical Section End ---
    return jsonify({"go": go_signal, "current_server_round": current_server_round}) # Optionally return server round


@app.route("/download", methods=["GET"])
def download_model():
    # Provide the latest aggregated model, regardless of which round a client thinks it's starting
    # Optionally, could add round check here too, but generally clients just need the newest model
    try:
        state_dict = model.state_dict()
        # You could also return the round number this model corresponds to (i.e., current_server_round - 1)
        return jsonify({
             "weights": encode_weights(state_dict),
             "model_round": current_server_round - 1 # The round that *produced* this model
             })
    except Exception as e:
        print(f"[✗] Error encoding model weights for download: {e}")
        return jsonify({"status": "error", "message": "Failed to encode model weights"}), 500

if __name__ == "__main__":
    # (Keep sys.path and log directory setup)
    log_dir = os.path.dirname(os.path.join(os.path.dirname(__file__), "accuracy_log.txt"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"--- Server starting ---")
    print(f"Expecting clients: {EXPECTED_CLIENTS}")
    print(f"Starting round: {current_server_round}")
    app.run(host="0.0.0.0", port=5000, debug=True) # Use debug=True for development