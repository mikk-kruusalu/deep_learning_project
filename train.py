import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
import torch
import yaml
from sklearn.metrics import f1_score

from cnn import SimpleCNN, UnetTransfer
from rnn import GRU, StackedLSTM, SentimentTransfer
from gnn import EnzymesGNN

models = {
    "simplecnn": SimpleCNN,
    "unet": UnetTransfer,
    "gru": GRU,
    "stacked_lstm": StackedLSTM,
    "sentiment_transfer": SentimentTransfer,
    "gnn_enzymes": EnzymesGNN,
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
    d = torch.load(file_path, weights_only=False, map_location="cpu")

    model = models[model_name]()
    model.load_state_dict(d["model"])

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(d["optimizer"])

    metrics = TrainingMetrics(**d["metrics"])

    return model, optimizer, metrics


def train(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    output_dir,
    nepochs=60,
    device="cpu",
    log_wandb=False,
    data_classes=None,
):
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
                labels.detach().cpu(), torch.argmax(pred.detach().cpu(), dim=-1), average="weighted"
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
                labels.detach().cpu(), torch.argmax(pred.detach().cpu(), dim=-1), average="weighted"
            ) / len(test_loader)

        metrics.test_losses.append(test_loss)
        metrics.test_f1s.append(test_f1)

        if epoch % 10 == 0 or epoch == (nepochs - 1):
            save_model(model, optimizer, metrics, Path(output_dir) / f"chk_{epoch}.pth")
        if log_wandb:
            log = {"test_loss": test_loss, "test_f1": test_f1, "train_loss": train_loss, "train_f1": train_f1}
            if epoch % 10 == 0 or epoch == (nepochs - 1):
                log.update(
                    {
                        "cm_epoch": epoch,
                        "train_cm": plot_confusion_matrix(train_loader, model, data_classes, device),
                        "test_cm": plot_confusion_matrix(test_loader, model, data_classes, device),
                    }
                )

            wandb.log(log)

        if epoch % 5 == 0 or epoch == (nepochs - 1):
            print(
                f"{epoch}: Test loss {metrics.test_losses[-1]:.4f}\t f1: {metrics.test_f1s[-1]:.3f} \t "
                f"Train loss: {metrics.train_losses[-1]:.3f}\t f1: {metrics.train_f1s[-1]:.3f}"
            )

    if log_wandb:
        wandb.finish()

    return model, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output-dir", default="cnn/checkpoints")
    parser.add_argument("-d", "--device", default="cpu")

    args = parser.parse_args()
    return args


def init_run(config):
    wandb.init(project=config["logger"]["project"], name=config["exp_name"], config=config["hyperparams"])


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    hyperparams = config["hyperparams"]

    # save config file alongside with the model checkpoints
    checkpoint_dir = Path(args.output_dir) / config["exp_name"]
    checkpoint_dir.mkdir(parents=True)
    with open(checkpoint_dir / Path(args.config).name, "w") as f:
        yaml.dump(config, f)

    if config["logger"]["wandb"]:
        import wandb
        from evaluate import plot_confusion_matrix

        init_run(config)

    if config["model"] in ["unet", "simplecnn"]:
        from cnn.dataset import load_data, get_dataloaders
    elif config["model"] in ["gnn_enzymes"]:
        from gnn.dataset import load_data, get_dataloaders
    else:
        from rnn.dataset import load_data, get_dataloaders

    train_data, test_data = load_data()
    train_loader, test_loader = get_dataloaders(train_data, test_data, batch_size=hyperparams["batch_size"])

    if "model" in hyperparams.keys():
        model = models[config["model"]](**hyperparams["model"])
    else:
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
        config["logger"]["wandb"],
        train_data.classes,
    )
