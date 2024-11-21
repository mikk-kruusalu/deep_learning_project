import torch.nn as nn


class StackedLSTM(nn.Module):
    def __init__(self):
        super(StackedLSTM, self).__init__()
