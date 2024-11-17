import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, inchannels=1, output_size=4):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
        )

        self.SEBlock1 = nn.Sequential(
            nn.AvgPool2d(59, 59, padding=0),
            nn.Flatten(start_dim=-3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Softmax(dim=-1),
            nn.Unflatten(-1, (64, 1, 1)),
        )

        self.sepconv = nn.Conv2d(64, 128, 1)

        self.module1 = nn.Sequential(nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.Conv2d(128, 128, 1))

        self.final = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Conv2d(128, 128, 5, 2),
            nn.ReLU(),
            nn.Flatten(start_dim=-3),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x * self.SEBlock1(x)

        # skip connection
        x = F.relu(self.sepconv(x) + self.module1(x))

        x = self.final(x)

        return x


# test if the model architecture works
if __name__ == "__main__":
    from dataset import load_data
    from torch.utils.data import DataLoader

    data, _ = load_data()

    train_loader = DataLoader(data, batch_size=64)

    model = SimpleCNN()
    for img, label in train_loader:
        print(model(img).shape)
        print(label.shape)
        break
