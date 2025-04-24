import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
sns.set(style="whitegrid")

# File paths
client_logs = {
    "6001": Path("client/6001_log.csv"),
    "6002": Path("client/6002_log.csv")
}
global_log = Path("server/global_log.csv")

# Label map
labels = {
    "6001": "Client 6001 (Digits 0–4)",
    "6002": "Client 6002 (Digits 5–9)"
}

# Function to plot train vs val metrics
def plot_train_val(df, model_label, metric, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df[f"train_{metric}"], label="Train")
    plt.plot(df["round"], df[f"val_{metric}"], label="Validation")
    plt.title(f"{model_label} — {metric.capitalize()} Curve")
    plt.xlabel("Federated Round")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    # save the plot
    plt.savefig(f"results/{model_label}_{metric}.png")
    plt.show()

# Function to plot global model test metrics
def plot_global(df):
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["loss"], label="Test Loss")
    plt.title("Global Model — Test Loss Curve")
    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.tight_layout()
    # save the plot
    plt.savefig("results/global_loss.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["accuracy"], label="Test Accuracy")
    plt.title("Global Model — Test Accuracy Curve")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    # save the plot
    plt.savefig("results/global_accuracy.png")
    plt.show()

# Read and plot for each client
for client_id, path in client_logs.items():
    if path.exists():
        df = pd.read_csv(path)
        df["round"] = df["round"].astype(int)
        plot_train_val(df, labels[client_id], "loss", "Loss")
        plot_train_val(df, labels[client_id], "acc", "Accuracy (%)")

# Global plot
if global_log.exists():
    df_global = pd.read_csv(global_log)
    df_global["round"] = df_global["round"].astype(int)
    plot_global(df_global)
