from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
import numpy as np
import os
import logging


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, eval_batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, eval_batch_size,
                         shuffle, validation_split, num_workers)


class GenderDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir='data', dataset='Adience', training=True,
                 test_cross_val=0):
        logging.info(f"test cross val is {test_cross_val}")
        if dataset == 'Adience':
            data = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                           allow_pickle=True).item()
            if training:
                data = [data[i] for i in range(5) if i != test_cross_val]
                data = [d for da in data for d in da]
            else:
                data = data[test_cross_val]
        else:
            raise NotImplementedError

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]['embedding']
        y = {'m': 0, 'f': 1}[self.data[idx]['gender']]

        return x, y


class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir='data', dataset='Adience', training=True,
                 test_cross_val=0):
        logging.info(f"test cross val is {test_cross_val}")
        if dataset == 'Adience':
            data = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                           allow_pickle=True).item()
            if training:
                data = [data[i] for i in range(5) if i != test_cross_val]
                data = [d for da in data for d in da]
            else:
                data = data[test_cross_val]
        else:
            raise NotImplementedError

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]['embedding']
        y = {1.0: 0, 5.0: 1, 10.0: 2, 17.5: 3, 28.5: 4,
             40.5: 5, 50.5: 6, 80.0: 7}[self.data[idx]['age']]

        return x, y


class GenderDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, eval_batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, dataset='Adience',
                 test_cross_val=0, training=True):

        self.dataset = GenderDataset(data_dir=data_dir, dataset=dataset,
                                     test_cross_val=test_cross_val,
                                     training=training)

        super().__init__(self.dataset, batch_size, eval_batch_size,
                         shuffle, validation_split, num_workers)


class AgeDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, eval_batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, dataset='Adience',
                 test_cross_val=0, training=True):

        self.dataset = AgeDataset(data_dir=data_dir, dataset=dataset,
                                  test_cross_val=test_cross_val,
                                  training=training)

        super().__init__(self.dataset, batch_size, eval_batch_size,
                         shuffle, validation_split, num_workers)
