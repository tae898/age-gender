import torch


def accuracy(output, target):
    """
    Vanilla accuracy: TP + TN / (TP + TN + FP + FN)
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def accuracy_mse(output, target):
    """
    This function was made when I was tinkering with regression rather than 
    classification. Just ignore it.
    """
    with torch.no_grad():
        assert len(output) == len(target)
        correct = 0
        correct += torch.sum(((output-target).abs() < 1)).item()
    return correct / len(target)


def accuracy_relaxed(output, target):
    """
    I made this function so that 101 age classes correspond to 8 age classes,
    for Adience dataset. Turns out this results in the same value as the vanilla 
    accuracy. Just ignore it. 

    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        correct = 0
        for p, t in zip(pred, target):
            if (0 <= p < 3) and (0 <= t < 3):
                correct += 1
            elif (3 <= p < 7) and (3 <= t < 7):
                correct += 1
            elif (7 <= p < 13.5) and (7 <= t < 13.5):
                correct += 1
            elif (13.5 <= p < 22.5) and (13.5 <= t < 22.5):
                correct += 1
            elif (22.5 <= p < 35) and (22.5 <= t < 35):
                correct += 1
            elif (35 <= p < 45.5) and (35 <= t < 45.5):
                correct += 1
            elif (45.5 <= p < 56.5) and (45.5 <= t < 56.5):
                correct += 1
            elif (56.5 <= p <= 100) and (56.5 <= t <= 100):
                correct += 1
            else:
                pass
    return correct / len(target)


def top_k_acc(output, target, k=3):
    """
    Not a useful metric for gender or age. Just ignore it.
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
