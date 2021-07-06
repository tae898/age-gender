import torch

# def accuracy_8(output, target):
#     with torch.no_grad():
#         ages = torch.tensor([1.0, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0])

#         correct = 0
#         for pred, label in zip(output, target):
#             pred = ages[torch.argmin(torch.abs(pred - ages))]
#             correct += torch.sum(pred == label).item()

#     return correct / len(target)

# def accuracy_101(output, target):
#     with torch.no_grad():
#         ages = torch.tensor([i for i in range(101)])

#         correct = 0
#         for pred, label in zip(output, target):
#             pred = ages[torch.argmin(torch.abs(pred - ages))]
#             correct += torch.sum(pred == label).item()

#     return correct / len(target)


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
