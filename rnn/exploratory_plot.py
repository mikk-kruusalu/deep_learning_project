import matplotlib.pyplot as plt
from dataset import load_data

data, _ = load_data()
print(data[:][1][:, None].shape)
print(data.labels)
fig, axes = plt.subplots(1, 6, constrained_layout=True, figsize=(12, 6))

for i, ax in enumerate(axes):
    ax.scatter(data[:][0][:, :, i], data[:][1][:, None].repeat((1, 128)))
    ax.set_xlabel(data.labels[i])

axes[0].set_ylabel("Activity")
plt.savefig("activity_vs_feature.png")
plt.show()
