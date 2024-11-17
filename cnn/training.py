import os
from dataclasses import dataclass, field

import torch
from sklearn.metrics import f1_score


@dataclass
class TrainingMetrics:
    train_losses: list[float] = field(default_factory=list)
    train_f1s: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    test_f1s: list[float] = field(default_factory=list)


def save_model(model, optimizer, file_path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        file_path,
    )


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
            pred = model(img)

            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().detach() / len(train_loader)
            train_f1 += f1_score(
                labels.cpu().detach(), torch.argmax(pred.cpu().detach(), dim=-1), average="weighted"
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

            test_loss += loss.cpu().detach() / len(test_loader)
            test_f1 += f1_score(
                labels.cpu().detach(), torch.argmax(pred.cpu().detach(), dim=-1), average="weighted"
            ) / len(test_loader)

        metrics.test_losses.append(test_loss)
        metrics.test_f1s.append(test_f1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_model(model, optimizer, os.path.join(output_dir, f"chk_{epoch}.pth"))
            print(
                f"{epoch}: Test loss {metrics.test_losses[-1]:.4f}\t f1: {metrics.test_f1s[-1]:.3f} \t "
                f"Train loss: {metrics.train_losses[-1]:.3f}\t f1: {metrics.train_f1s[-1]:.3f}"
            )

    return model, metrics


# test if the training loop works
if __name__ == "__main__":
    from dataset import load_data
    from simplecnn import SimpleCNN
    from torch.utils.data import DataLoader

    train_data, test_data = load_data()

    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, criterion, optimizer, train_loader, test_loader, nepochs=1)
