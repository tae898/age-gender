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

def accuracy_relaxed(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        correct = 0
        for p, t in zip(pred, target):
            if (0 <= p <= 2) and (0 <= t <= 2):
                correct+=1
            elif (4 <= p <= 6) and (4 <= t <= 6):
                correct+=1
            elif (8 <= p <= 12) and (8 <= t <= 12):
                correct+=1
            elif (15 <= p <= 20) and (15 <= t <= 20):
                correct+=1
            elif (25 <= p <= 32) and (25 <= t <= 32):
                correct+=1
            elif (38 <= p <= 43) and (38 <= t <= 43):
                correct+=1
            elif (48 <= p <= 53) and (48 <= t <= 53):
                correct+=1
            elif (60 <= p <= 100) and (60 <= t <= 100):
                correct+=1
            else:
                pass
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
