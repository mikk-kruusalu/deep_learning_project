import torch.nn as nn
import torch.nn.functional as F


class StackedLSTM(nn.Module):
    def __init__(self, in_features=6, nclasses=6, nlayers=2):
        super(StackedLSTM, self).__init__()

        self.lstm = nn.LSTM(in_features, nclasses, batch_first=True, num_layers=nlayers)

    def forward(self, x):
        out, (h, c) = self.lstm(x)

        return F.softmax(h[-1], dim=-1)
