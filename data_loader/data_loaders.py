from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
import numpy as np
import os
import logging


class GenderDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir='data', dataset=None, training=True,
                 test_cross_val=0):
        logging.info(f"test cross val is {test_cross_val}")
        if dataset.lower() == 'adience':
            data = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                           allow_pickle=True).item()
            if training:
                data = [data[i] for i in range(5) if i != test_cross_val]
                data = [d for da in data for d in da]
            else:
                data = data[test_cross_val]

        elif dataset.lower() in ['wiki', 'imdb']:
            data = np.load(os.path.join(data_dir, f"{dataset.lower()}_crop/data.npy"),
                           allow_pickle=True)
        elif dataset.lower() == 'imdb_wiki':
            data_imdb = np.load(os.path.join(data_dir, f"imdb_crop/data.npy"),
                                allow_pickle=True).tolist()
            data_wiki = np.load(os.path.join(data_dir, f"wiki_crop/data.npy"),
                                allow_pickle=True).tolist()

            data = data_imdb + data_wiki

            del data_imdb, data_wiki
        elif dataset.lower() == 'imdb_wiki_adience':
            data_imdb = np.load(os.path.join(data_dir, f"imdb_crop/data.npy"),
                                allow_pickle=True).tolist()
            data_wiki = np.load(os.path.join(data_dir, f"wiki_crop/data.npy"),
                                allow_pickle=True).tolist()
            data_adience = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                                   allow_pickle=True).item()
            data_adience = [sample for _, samples in data_adience.items()
                            for sample in samples]

            data = data_imdb + data_wiki + data_adience

            del data_imdb, data_wiki, data_adience
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

    def __init__(self, data_dir='data', dataset=None, training=True,
                 test_cross_val=0, num_classes=8):
        logging.info(f"test cross val is {test_cross_val}")

        if num_classes == 8:
            self.age_map = {1.0: 0, 5.0: 1, 10.0: 2, 17.5: 3, 28.5: 4,
                            40.5: 5, 50.5: 6, 80.0: 7}
        elif num_classes == 101:
            self.age_map = {i: i for i in range(101)}
        else:
            raise NotImplementedError

        if dataset == 'Adience':
            data = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                           allow_pickle=True).item()

            if training:
                data = [data[i] for i in range(5) if i != test_cross_val]
                data = [d for da in data for d in da]
            else:
                data = data[test_cross_val]

        elif dataset.lower() in ['wiki', 'imdb']:
            data = np.load(os.path.join(data_dir, f"{dataset.lower()}_crop/data.npy"),
                           allow_pickle=True)
        elif dataset.lower() == 'imdb_wiki':
            data_imdb = np.load(os.path.join(data_dir, f"imdb_crop/data.npy"),
                                allow_pickle=True).tolist()
            data_wiki = np.load(os.path.join(data_dir, f"wiki_crop/data.npy"),
                                allow_pickle=True).tolist()

            data = data_imdb + data_wiki

            del data_imdb, data_wiki
        elif dataset.lower() == 'imdb_wiki_adience':
            data_imdb = np.load(os.path.join(data_dir, f"imdb_crop/data.npy"),
                                allow_pickle=True).tolist()
            data_wiki = np.load(os.path.join(data_dir, f"wiki_crop/data.npy"),
                                allow_pickle=True).tolist()
            data_adience = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                                   allow_pickle=True).item()
            data_adience = [sample for _, samples in data_adience.items()
                            for sample in samples]

            data = data_imdb + data_wiki + data_adience

            del data_imdb, data_wiki, data_adience
        else:
            raise NotImplementedError

        self.data = data

    def __len__(self):
        return len(self.data)

    def _get_closest_age(self, num):
        possible_ages = np.array([age for age in list(self.age_map.keys())])
        idx = int(np.argmin(np.abs(possible_ages - num)))

        return possible_ages[idx]

    def __getitem__(self, idx):
        x = self.data[idx]['embedding']
        y = self.age_map[self._get_closest_age(self.data[idx]['age'])]

        return x, y


class GenderDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split,
                 num_workers, dataset, num_classes, test_cross_val, training):

        self.dataset = GenderDataset(data_dir=data_dir, dataset=dataset,
                                     test_cross_val=test_cross_val,
                                     training=training)

        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)


class AgeDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split,
                 num_workers, dataset, num_classes, test_cross_val, training):

        self.dataset = AgeDataset(data_dir=data_dir, dataset=dataset,
                                  test_cross_val=test_cross_val,
                                  training=training, num_classes=num_classes)

        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)
