import torch.nn as nn
import torch


class UnetTransfer(nn.Module):
    def __init__(self, inchannels=1, output_size=4):
        super(UnetTransfer, self).__init__()

        self.unet = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )

        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.unet.bottleneck.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=8),
            nn.Flatten(start_dim=-3),
            nn.SELU(),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.unet.encoder1(x.repeat(1, 3, 1, 1))
        x = self.unet.pool1(x)
        x = self.unet.encoder2(x)
        x = self.unet.pool2(x)
        x = self.unet.encoder3(x)
        x = self.unet.pool3(x)
        x = self.unet.encoder4(x)
        x = self.unet.pool4(x)

        bottleneck = self.unet.bottleneck(x)

        out = self.classifier(bottleneck)

        return out


# test if the model architecture works
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import load_data

    data, _ = load_data()

    train_loader = DataLoader(data, batch_size=64)

    model = UnetTransfer()
    for img, label in train_loader:
        print(model(img).shape)
        print(label.shape)
        break
