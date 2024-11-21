import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
import torch
import yaml
from dataset import load_data
from torch.utils.data import DataLoader
from simplecnn import SimpleCNN
from unet_transfer import UnetTransfer
from sklearn.metrics import f1_score


models = {
    "simplecnn": SimpleCNN,
    "unet": UnetTransfer,
}


@dataclass
class TrainingMetrics:
    train_losses: list[float] = field(default_factory=list)
    train_f1s: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    test_f1s: list[float] = field(default_factory=list)


def save_model(model, optimizer, metrics, file_path):
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": asdict(metrics)},
        file_path,
    )


def load_model(model_name: str, file_path: Path):
    d = torch.load(file_path, weights_only=False)

    model = models[model_name]()
    model.load_state_dict(d["model"])

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(d["optimizer"])

    metrics = TrainingMetrics(**d["metrics"])

    return model, optimizer, metrics


def train(model, criterion, optimizer, train_loader, test_loader, output_dir, nepochs=60, device="cpu"):
    model = model.to(device)

    metrics = TrainingMetrics()
    for epoch in range(nepochs):
        model.train()
        train_loss = 0
        train_f1 = 0
        for img, labels in train_loader:
            img = img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            pred = model(img)
            loss = criterion(pred, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            train_f1 += f1_score(
                labels.item(), torch.argmax(pred.item(), dim=-1), average="weighted"
            ) / len(train_loader)

        metrics.train_losses.append(train_loss)
        metrics.train_f1s.append(train_f1)

        model.eval()
        test_loss = 0
        test_f1 = 0
        for img, labels in test_loader:
            img = img.to(device)
            labels = labels.to(device)
            pred = model(img)

            loss = criterion(pred, labels)

            test_loss += loss.item() / len(test_loader)
            test_f1 += f1_score(
                labels.item(), torch.argmax(pred.item(), dim=-1), average="weighted"
            ) / len(test_loader)

        metrics.test_losses.append(test_loss)
        metrics.test_f1s.append(test_f1)

        if epoch % 5 == 0:
            save_model(model, optimizer, metrics, Path(output_dir) / f"chk_{epoch}.pth")
            print(
                f"{epoch}: Test loss {metrics.test_losses[-1]:.4f}\t f1: {metrics.test_f1s[-1]:.3f} \t "
                f"Train loss: {metrics.train_losses[-1]:.3f}\t f1: {metrics.train_f1s[-1]:.3f}"
            )

    save_model(model, optimizer, metrics, Path(output_dir) / f"chk_{epoch}.pth")

    return model, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output-dir", default="cnn/checkpoints")
    parser.add_argument("-d", "--device", default="cpu")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    hyperparams = config["hyperparams"]

    # save config file alongside with the model checkpoints
    checkpoint_dir = Path(args.output_dir) / config["exp_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / Path(args.config).name, "w") as f:
        yaml.dump(config, f)

    train_data, test_data = load_data()
    train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"])

    model = models[config["model"]]()

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    train(
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        checkpoint_dir,
        hyperparams["nepochs"],
        config["device"],
    )
