# server/plot_accuracy.py

import matplotlib.pyplot as plt
import os

LOG_FILE = "server/accuracy_log.txt"
PLOT_FILE = "plots/accuracy_plot.png"

# === Load log data ===
rounds = []
accuracies = []

with open(LOG_FILE, "r") as f:
    for line in f:
        r, acc = line.strip().split(",")
        rounds.append(int(r))
        accuracies.append(float(acc))

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(rounds, accuracies, marker='o')
plt.title("Federated Learning Accuracy Over Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.ylim(0, 100)
plt.xticks(rounds)

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)
plt.savefig(PLOT_FILE)
plt.show()
