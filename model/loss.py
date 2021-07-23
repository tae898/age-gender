import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cse(output, target):
    """
    Using the pytorch cross entropy loss means that you don't have to add a 
    softmax activation.

    """
    return F.cross_entropy(output, target)


def mse(output, target):
    return F.mse_loss(output, target)
