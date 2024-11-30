import matplotlib.pyplot as plt
from dataset import load_data
import torch

data, _ = load_data()
print(data.seq.shape)
print(data.y.shape)
print(data.labels)

for i, label in enumerate(data.labels):
    mean = data.seq[:, :, i].mean().item()
    std = data.seq[:, :, i].std().item()
    pc20 = torch.quantile(data.seq[:, :, i], 0.20).item()
    pc80 = torch.quantile(data.seq[:, :, i], 0.80).item()
    print(
        f"{label}:\t mean: {round(mean, 4)} \t std: {round(std, 4)} \t"
        f"pc20: {round(pc20, 4)} \t pc80: {round(pc80, 4)}"
    )

fig, axes = plt.subplots(1, 6, constrained_layout=True, figsize=(12, 6))

for i, ax in enumerate(axes):
    ax.scatter(data.seq[:, :, i], data.y[:, None].repeat((1, 128)))
    ax.set_xlabel(data.labels[i])

axes[0].set_ylabel("Activity")
plt.savefig("figures/activity_vs_feature.png")
