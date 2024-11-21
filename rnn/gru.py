import torch.nn as nn
import torch


# https://d2l.ai/chapter_recurrent-modern/gru.html
class GRUnit(nn.Module):
    def __init__(self, nhidden=128, nfeatures=6):
        super(GRUnit, self).__init__()

        self.reset = nn.Sequential(
            nn.Linear(nhidden + nfeatures, nhidden),
            nn.Sigmoid(),
        )

        self.update = nn.Sequential(
            nn.Linear(nhidden + nfeatures, nhidden),
            nn.Sigmoid(),
        )

        self.candidates = nn.Sequential(
            nn.Linear(nhidden + nfeatures, nhidden),
            nn.Tanh(),
        )

    def forward(self, H_prev, X):
        # H_prev: [num_examples, num_hidden]
        # X: [num_examples, num_features]

        cat = torch.cat((H_prev, X), dim=-1)
        R = self.reset(cat)
        Z = self.update(cat)

        candidates = self.candidates(torch.cat((R * H_prev, X), dim=-1))

        H = Z * H_prev + (1 - Z) * candidates

        return H


class GRU(nn.Module):
    def __init__(self, nhidden=128, nfeatures=6, nclasses=6, dropout=0.5):
        super(GRU, self).__init__()

        self.nhidden = nhidden
        self.gru = GRUnit(nhidden, nfeatures)
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
        # x: [batches, sequence, features]

        H = torch.zeros((x.shape[0], self.nhidden), device=x.device)
        for seq in x.permute(1, 0, 2):
            H = self.gru(H, seq)

        probs = self.classifier(H)
        return probs


if __name__ == "__main__":
    from dataset import load_data
    from torch.utils.data import DataLoader

    data, _ = load_data()

    train_loader = DataLoader(data, batch_size=64)

    model = GRU()

    for seq, label in train_loader:
        print(model(seq).shape)
        print(label.shape)
        break
