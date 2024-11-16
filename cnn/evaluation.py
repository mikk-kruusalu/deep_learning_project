from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from training import TrainingMetrics


def plot_learning_curves(metrics: TrainingMetrics):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(metrics.test_losses, label="test")
    ax[0].plot(metrics.train_losses, label="train")
    ax[0].set_title("Losses")
    ax[0].legend()

    ax[1].plot(metrics.test_f1s, label="test")
    ax[1].plot(metrics.train_f1s, label="train")
    ax[1].set_title("F1 score")
    ax[1].legend()

    return fig


def plot_confusion_matrix(loader, model, classes, device="cpu", ax=None, **fig_kw):
    test_pred = []
    test_labels = []
    for seq, label in loader:
        seq = seq.to(device)
        test_pred.append(model(seq).detach().cpu())
        test_labels.append(label)

    test_pred = torch.concatenate(test_pred)
    test_labels = torch.concatenate(test_labels)
    test_pred = torch.argmax(test_pred, dim=-1)

    cm_display = ConfusionMatrixDisplay(confusion_matrix(test_labels, test_pred), display_labels=classes)

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, **fig_kw)
    cm_display.plot(xticks_rotation=45, ax=ax)


# test plotting functions
if __name__ == "__main__":
    metrics = TrainingMetrics()
    metrics.train_losses = torch.rand(10).tolist()
    metrics.test_losses = torch.rand(10).tolist()
    metrics.train_f1s = torch.rand(10).tolist()
    metrics.test_f1s = torch.rand(10).tolist()

    plot_learning_curves(metrics)
    plt.show()

    from torch.utils.data import DataLoader
    from dataset import load_data
    from simplecnn import SimpleCNN

    _, data = load_data()
    loader = DataLoader(data, batch_size=64)

    model = SimpleCNN()
    plot_confusion_matrix(loader, model, data.classes)
    plt.show()
