import torch.nn.functional as F
import torch


def nll_loss(output: torch.tensor, target: torch.tensor):
    return F.nll_loss(output, target)


def cse(output: torch.tensor, target: torch.tensor):
    """
    Using the pytorch cross entropy loss means that you don't have to add a 
    softmax activation.

    """
    return F.cross_entropy(output, target)


def mse(output: torch.tensor, target: torch.tensor):
    return F.mse_loss(output, target)
