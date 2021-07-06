import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MLPGender(BaseModel):
    def __init__(self, dropout=0.5, num_layers=None, num_classes=None):
        super().__init__()

        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, 8)
        self.linear7 = nn.Linear(8, 4)
        self.linear8 = nn.Linear(4, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)

        x = F.relu(self.linear2(x))
        x = self.dropout(x)

        x = F.relu(self.linear3(x))
        x = self.dropout(x)

        x = F.relu(self.linear4(x))
        x = self.dropout(x)

        x = F.relu(self.linear5(x))
        x = self.dropout(x)

        x = F.relu(self.linear6(x))
        x = self.dropout(x)

        x = F.relu(self.linear7(x))
        x = self.dropout(x)

        x = self.linear8(x)

        return x


class MLPAge(BaseModel):
    def __init__(self, dropout=0.5, num_layers=None, num_classes=None):
        super().__init__()

        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)

        x = F.relu(self.linear2(x))
        x = self.dropout(x)

        x = F.relu(self.linear3(x))
        x = self.dropout(x)

        x = F.relu(self.linear4(x))
        x = self.dropout(x)

        x = F.relu(self.linear5(x))
        x = self.dropout(x)

        x = self.linear6(x)

        return x


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
        first_block = []
        first_block.append(Residual(in_features, dropout, ic_beginning))
        for _ in range(num_residuals_per_block-1):
            first_block.append(Residual(in_features, dropout, True))
        first_block.append(DownSample(
            in_features, in_features//2, dropout))

        return first_block

    def forward(self, x):
        return self.blocks(x)
