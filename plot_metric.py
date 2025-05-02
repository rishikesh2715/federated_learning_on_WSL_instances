# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# from pathlib import Path
# import os

# # === Font Setup ===
# plt.rcParams.update({
#     'font.family': 'Courier New',
#     'font.size': 20,
#     'axes.titlesize': 20,
#     'axes.labelsize': 20,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20,
#     'legend.fontsize': 20,
#     'figure.titlesize': 20
# })

# def apply_style(ax):
#     ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.45)
#     ax.minorticks_on()
#     ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.35)
#     ax.set_axisbelow(True)

# # === File Paths ===
# client_logs = {
#     "6001": Path("client/6001_log.csv"),
#     "6002": Path("client/6002_log.csv")
# }
# global_log = Path("server/global_log.csv")

# labels = {
#     "6001": "Pi1",
#     "6002": "Pi2"
# }

# # === Colors ===
# color_map = {
#     "6001_train": "#379392",
#     "6001_val": "#4FB0C6",
#     "6002_train": "#4F86C6",
#     "6002_val": "#744FC6"
# }

# # === Load and Prepare Data ===
# dfs = {}
# for cid, path in client_logs.items():
#     if path.exists():
#         df = pd.read_csv(path)
#         df["round"] = df["round"].astype(int)
#         # Optional clipping to avoid wild outliers (comment out if not needed)
#         df["train_loss"] = df["train_loss"].clip(lower=0, upper=5)
#         df["val_loss"] = df["val_loss"].clip(lower=0, upper=5)
#         df["train_acc"] = df["train_acc"].clip(lower=0, upper=100)
#         df["val_acc"] = df["val_acc"].clip(lower=0, upper=100)
#         dfs[cid] = df

# # === Combined Client Loss Plot ===
# fig, ax = plt.subplots(figsize=(8, 8))
# for cid in dfs:
#     df = dfs[cid]
#     ax.plot(df["round"], df["train_loss"], label=f"{labels[cid]} — Train", color=color_map[f"{cid}_train"], linestyle="-", linewidth=2)
#     ax.plot(df["round"], df["val_loss"], label=f"{labels[cid]} — Validation", color=color_map[f"{cid}_val"], linestyle="--", linewidth=2)

# ax.set_xlabel("Federated Round")
# ax.set_ylabel("Loss")
# ax.set_title("Client Loss Curves")
# apply_style(ax)

# # Legend on top outside
# handles, labels_text = ax.get_legend_handles_labels()
# ax.legend(handles, labels_text, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

# plt.tight_layout()
# plt.savefig("results/combined_client_loss.pdf")
# plt.show()

# # === Combined Client Accuracy Plot ===
# fig, ax = plt.subplots(figsize=(8, 8))
# for cid in dfs:
#     df = dfs[cid]
#     ax.plot(df["round"], df["train_acc"], label=f"{labels[cid]} — Train", color=color_map[f"{cid}_train"], linestyle="-", linewidth=2)
#     ax.plot(df["round"], df["val_acc"], label=f"{labels[cid]} — Validation", color=color_map[f"{cid}_val"], linestyle="--", linewidth=2)

# ax.set_xlabel("Federated Round")
# ax.set_ylabel("Accuracy (%)")
# ax.set_title("Client Accuracy Curves")
# apply_style(ax)

# handles, labels_text = ax.get_legend_handles_labels()
# ax.legend(handles, labels_text, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

# plt.tight_layout()
# plt.savefig("results/combined_client_accuracy.pdf")
# plt.show()

# # === Global Model Plot ===
# if global_log.exists():
#     df_global = pd.read_csv(global_log)
#     df_global["round"] = df_global["round"].astype(int)

#     fig, ax1 = plt.subplots(figsize=(8, 8))
#     ax2 = ax1.twinx()

#     ax1.plot(df_global["round"], df_global["loss"], label="Global Loss", color="#9671bd", linewidth=2)
#     ax2.plot(df_global["round"], df_global["accuracy"], label="Global Accuracy", color="#77b5b6", linewidth=2)

#     ax1.set_xlabel("Federated Round")
#     ax1.set_ylabel("Loss")
#     ax2.set_ylabel("Accuracy (%)")
#     ax1.set_title("Global Model — Loss & Accuracy")

#     apply_style(ax1)
#     apply_style(ax2)

#     # Combine legends manually
#     lines = ax1.get_lines() + ax2.get_lines()
#     labels = [l.get_label() for l in lines]
#     ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

#     plt.tight_layout()
#     plt.savefig("results/global_combined.pdf")
#     plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from matplotlib.ticker import MultipleLocator

# === Font Setup ===
plt.rcParams.update({
    'font.family': 'Courier New',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})

def apply_style(ax):
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)

# === File Paths ===
client_logs = {
    "6001": Path("client/pre_6001_log.csv"),
    "6002": Path("client/pre_6002_log.csv")
}
global_log = Path("server/global_log_pre.csv")

labels = {
    "6001": "Pi1",
    "6002": "Pi2"
}

# === Colors (two only) ===
pi_colors = {
    "6001": "#4FB0C6",
    "6002": "#744FC6"
}

# === Load and Prepare Data ===
dfs = {}
for cid, path in client_logs.items():
    if path.exists():
        df = pd.read_csv(path)
        df["round"] = df["round"].astype(int)
        # Optional clipping to avoid wild outliers
        df["train_loss"] = df["train_loss"].clip(lower=0, upper=5)
        df["val_loss"]   = df["val_loss"].clip(lower=0, upper=5)
        df["train_acc"]  = df["train_acc"].clip(lower=0, upper=100)
        df["val_acc"]    = df["val_acc"].clip(lower=0, upper=100)
        dfs[cid] = df

# === Combined Client Loss Plot ===
fig, ax = plt.subplots(figsize=(10, 10))
for cid, df in dfs.items():
    color = pi_colors[cid]
    ax.plot(df["round"], df["train_loss"],
            label=f"{labels[cid]} — Train",
            color=color, linestyle="--", linewidth=2)
    ax.plot(df["round"], df["val_loss"],
            label=f"{labels[cid]} — Validation",
            color=color, linestyle="-", linewidth=2)

ax.set_xlim(0, 20) # include round 0
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.set_xlabel("Federated Round")
ax.set_ylabel("Loss")
ax.set_title("Client Loss Curves")
apply_style(ax)

# Legend inside
ax.legend(loc='upper right', framealpha=0.5)

plt.tight_layout()
plt.savefig("results/combined_client_loss.pdf")
plt.show()

# === Combined Client Accuracy Plot ===
fig, ax = plt.subplots(figsize=(10, 10))
for cid, df in dfs.items():
    color = pi_colors[cid]
    ax.plot(df["round"], df["train_acc"],
            label=f"{labels[cid]} — Train",
            color=color, linestyle="--", linewidth=2)
    ax.plot(df["round"], df["val_acc"],
            label=f"{labels[cid]} — Validation",
            color=color, linestyle="-", linewidth=2)

ax.set_xlim(0, 20)  # include round 0
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.set_xlabel("Federated Round")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Client Accuracy Curves")
apply_style(ax)

ax.legend(loc='lower right', framealpha=0.5)

plt.tight_layout()
plt.savefig("results/combined_client_accuracy.pdf")
plt.show()

# === Global Model Plot (optional, same style) ===
if global_log.exists():
    df_global = pd.read_csv(global_log)
    df_global["round"] = df_global["round"].astype(int)

    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()

    ax1.plot(df_global["round"], df_global["loss"],
             label="Global Loss", color="#9671bd", linewidth=2, linestyle='-')
    ax2.plot(df_global["round"], df_global["accuracy"],
             label="Global Accuracy", color="#77b5b6", linewidth=2, linestyle='-')

    ax1.set_xlim(0,20)
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy (%)")
    ax1.set_title("Global Model — Loss & Accuracy")

    apply_style(ax1)
    apply_style(ax2)

    # Combined legend inside
    lines = ax1.get_lines() + ax2.get_lines()
    labels_ = [l.get_label() for l in lines]
    ax1.legend(lines, labels_, loc='center right', framealpha=0.5)

    plt.tight_layout()
    plt.savefig("results/global_combined.pdf")
    plt.show()
