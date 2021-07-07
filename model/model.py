import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class Residual(nn.Module):
    def __init__(self, num_features, dropout, ic_beginning=False):
        super().__init__()
        self.num_features = num_features
        self.ic_beginning = ic_beginning

        if self.ic_beginning:
            self.norm_layer1 = nn.BatchNorm1d(num_features)
            self.dropout1 = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(num_features, num_features)
        self.relu1 = nn.ReLU()

        self.norm_layer2 = nn.BatchNorm1d(num_features)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(num_features, num_features)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        identity = out = x

        if self.ic_beginning:
            out = self.norm_layer1(x)
            out = self.dropout1(out)

        out = self.linear1(out)
        out = self.relu1(out)

        out = self.norm_layer2(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        out += identity
        out = self.relu2(out)

        return out


class DownSample(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()

        assert in_features > out_features

        self.in_features = in_features
        self.out_features = out_features

        self.norm_layer = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        out = self.norm_layer(out)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.relu(out)

        return out


class ResMLP(BaseModel):
    def __init__(self, dropout, num_residuals_per_block, num_blocks, num_classes,
                 num_initial_features):
        super().__init__()

        blocks = []
        blocks.extend(self._create_block(
            num_initial_features, dropout, num_residuals_per_block, False))
        num_initial_features //= 2

        for _ in range(num_blocks-1):
            blocks.extend(self._create_block(
                num_initial_features, dropout, num_residuals_per_block, True))
            num_initial_features //= 2

        blocks.append(nn.Linear(num_initial_features, num_classes))

        self.blocks = nn.Sequential(*blocks)

    def _create_block(self, in_features, dropout, num_residuals_per_block, ic_beginning):
        block = []
        if num_residuals_per_block > 0:
            block.append(Residual(in_features, dropout, ic_beginning))
        for _ in range(num_residuals_per_block-1):
            block.append(Residual(in_features, dropout, True))
        block.append(DownSample(
            in_features, in_features//2, dropout))

        return block

    def forward(self, x):
        return self.blocks(x)
