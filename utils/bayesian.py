import torch
from scipy.stats import entropy as calc_entropy
from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def forward_mc(model: torch.nn.Module, embedding: np.ndarray,
               num_passes: int = 512) -> tuple:
    """Perform MC dropout

    https://arxiv.org/pdf/1506.02142.pdf
    """

    # ArcFace is 512-D.
    assert embedding.shape == (1, 512)
    embedding = torch.tensor(embedding)
    embedding = embedding.repeat(num_passes, 1)
    with torch.no_grad():
        if device == 'cuda:0':
            with autocast():
                outputs = model(embedding)
        else:
            outputs = model(embedding)

        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.detach().cpu().numpy()

    preds = np.argmax(outputs, axis=1)
    expectation = preds.mean().item()

    value, counts = np.unique(preds, return_counts=True)
    entropy = calc_entropy(counts)
    entropy = entropy.item()

    # expectation is the approximation of the bayesian posterior prediction,
    # and entorpy is the measure of uncertainty. 0 means certain.

    # MAXIMUM_ENTROPY = {'gender': 0.6931471805599453,
    #                    'age': 4.615120516841261}

    return expectation, entropy


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            print(m)
