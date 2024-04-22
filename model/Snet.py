import torch.nn as nn


class Snet(nn.Module):
    def __init__(self, in_features, out_features=10):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x
