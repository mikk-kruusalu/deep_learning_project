import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from train import TrainingMetrics, load_model
import argparse
from pathlib import Path
from torch.utils.data import DataLoader


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
    model.eval()
    test_pred = []
    test_labels = []
    for img, label in loader:
        img = img.to(device)
        test_pred.append(model(img).detach().cpu())
        test_labels.append(label)

    test_pred = torch.concatenate(test_pred)
    test_labels = torch.concatenate(test_labels)
    test_pred = torch.argmax(test_pred, dim=-1)

    cm_display = ConfusionMatrixDisplay(confusion_matrix(test_labels, test_pred), display_labels=classes)

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, **fig_kw)
    cm_display.plot(xticks_rotation=45, ax=ax)

    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch_size", default=64)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    file = Path(args.file)

    model, _, metrics = load_model(args.model, file)

    plot_learning_curves(metrics)
    plt.savefig(file.parent / f"{file.stem}_learning_curves.png")

    if args.model in ["unet", "simplecnn"]:
        from cnn.dataset import load_data
    else:
        from rnn.dataset import load_data

    _, test_data = load_data()
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    plot_confusion_matrix(test_loader, model, test_data.classes, args.device)
    plt.savefig(file.parent / f"{file.stem}_confusion_matrix.png")
