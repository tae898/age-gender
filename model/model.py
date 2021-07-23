from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class LinearBounded(nn.Module):
    """
    This custom activation function was made so that the output is always 
    bounded within the specified range. In the range, it's just linear.

    In the end I didn't use this since regression results were worse than 
    classification results.
    """

    def __init__(self, min_bound, max_bound):
        super().__init__()

        assert min_bound < max_bound

        self.min_bound = min_bound
        self.max_bound = max_bound

    def forward(self, x):

        return torch.clamp(x, min=self.min_bound, max=self.max_bound)


class SigmoidBounded(nn.Module):
    """
    This custom activation function was made so that I can specify the
    minimum and maximum of the sigmoid function.

    In the end I didn't use this since regression results were worse than 
    classification results.

    """

    def __init__(self, min_bound, max_bound):
        super().__init__()

        assert min_bound < max_bound

        self.min_bound = min_bound
        self.max_bound = max_bound

    def forward(self, x):

        return torch.sigmoid(x) * (self.max_bound - self.min_bound) + self.min_bound


class Residual(nn.Module):
    """
    This module looks like what you find in the original resnet or IC paper     
    (https://arxiv.org/pdf/1905.05928.pdf), except that it's based on MLP, not CNN. 
    If you flag `only_MLP` as True, then it won't use any batch norm, dropout, or
    residual connections 

    """

    def __init__(self, num_features, dropout, ic_beginning=False, only_MLP=False):
        super().__init__()
        self.num_features = num_features
        self.ic_beginning = ic_beginning
        self.only_MLP = only_MLP

        if self.ic_beginning and (not self.only_MLP):
            self.norm_layer1 = nn.BatchNorm1d(num_features)
            self.dropout1 = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(num_features, num_features)
        self.relu1 = nn.ReLU()

        if not self.only_MLP:
            self.norm_layer2 = nn.BatchNorm1d(num_features)
            self.dropout2 = nn.Dropout(p=dropout)

        self.linear2 = nn.Linear(num_features, num_features)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        identity = out = x

        if self.ic_beginning and (not self.only_MLP):
            out = self.norm_layer1(x)
            out = self.dropout1(out)

        out = self.linear1(out)
        out = self.relu1(out)

        if not self.only_MLP:
            out = self.norm_layer2(out)
            out = self.dropout2(out)

        out = self.linear2(out)

        if not self.only_MLP:
            out += identity

        out = self.relu2(out)

        return out


class DownSample(nn.Module):
    """
    This module is an MLP, where the number of output features is lower than 
    that of input features. If you flag `only_MLP` as False, it'll add norm 
    and dropout 

    """

    def __init__(self, in_features, out_features, dropout, only_MLP=False):
        super().__init__()
        assert in_features > out_features

        self.in_features = in_features
        self.out_features = out_features
        self.only_MLP = only_MLP

        if not self.only_MLP:
            self.norm_layer = nn.BatchNorm1d(in_features)
            self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x

        if not self.only_MLP:
            out = self.norm_layer(out)
            out = self.dropout(out)

        out = self.linear(out)
        out = self.relu(out)

        return out


class ResMLP(BaseModel):
    """
    MLP with optinally batch norm, dropout, and residual connections. I got 
    inspiration from the original ResNet paper and https://arxiv.org/pdf/1905.05928.pdf.

    Downsampling is done after every block so that the features can be encoded 
    and compressed.

    """

    def __init__(self, dropout, num_residuals_per_block, num_blocks, num_classes,
                 num_initial_features, last_activation=None, min_bound=None,
                 max_bound=None, only_MLP=False):
        super().__init__()

        blocks = []
        blocks.extend(self._create_block(
            num_initial_features, dropout, num_residuals_per_block, False, only_MLP=only_MLP))
        num_initial_features //= 2

        for _ in range(num_blocks-1):
            blocks.extend(self._create_block(
                num_initial_features, dropout, num_residuals_per_block, True, only_MLP=only_MLP))
            num_initial_features //= 2

        blocks.append(nn.Linear(num_initial_features, num_classes))

        if last_activation == 'LinearBounded':
            blocks.append(LinearBounded(
                min_bound=min_bound, max_bound=max_bound))
        elif last_activation == 'SigmoidBounded':
            blocks.append(SigmoidBounded(
                min_bound=min_bound, max_bound=max_bound))

        self.blocks = nn.Sequential(*blocks)

    def _create_block(self, in_features, dropout, num_residuals_per_block, ic_beginning, only_MLP):
        block = []
        if num_residuals_per_block > 0:
            block.append(Residual(in_features, dropout,
                                  ic_beginning, only_MLP=only_MLP))
        for _ in range(num_residuals_per_block-1):
            block.append(Residual(in_features, dropout,
                                  True, only_MLP=only_MLP))
        block.append(DownSample(
            in_features, in_features//2, dropout, only_MLP=only_MLP))

        return block

    def forward(self, x):
        return self.blocks(x)
