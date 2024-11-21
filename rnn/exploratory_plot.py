import matplotlib.pyplot as plt
from dataset import load_data

data, _ = load_data()
print(data.seq.shape)
print(data.labels)
fig, axes = plt.subplots(1, 6, constrained_layout=True, figsize=(12, 6))

for i, ax in enumerate(axes):
    ax.scatter(data.seq[:, :, i], data.y[:, None].repeat((1, 128)))
    ax.set_xlabel(data.labels[i])

axes[0].set_ylabel("Activity")
plt.savefig("activity_vs_feature.png")
plt.show()
