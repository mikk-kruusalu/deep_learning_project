import torch.nn as nn
import torch.nn.functional as F
import torch


class StackedLSTM(nn.Module):
    def __init__(self, in_features=2, nclasses=6, nhidden=128, nlayers=2, dropout=0.2):
        super(StackedLSTM, self).__init__()

        self.lstm = nn.LSTM(in_features, nhidden, batch_first=True, num_layers=nlayers)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, nclasses),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        acc = torch.norm(x[:, :, :3], dim=-1)
        gyro = torch.norm(x[:, :, 3:], dim=-1)
        x = torch.stack((acc, gyro), dim=-1)
        out, (h, c) = self.lstm(x)

        classes = self.classifier(h[-1])

        return F.softmax(classes, dim=-1)


if __name__ == "__main__":
    from dataset import load_data
    from torch.utils.data import DataLoader

    data, _ = load_data()

    train_loader = DataLoader(data, batch_size=64)

    model = StackedLSTM()

    for seq, label in train_loader:
        print(model(seq).shape)
        print(label.shape)
        break
