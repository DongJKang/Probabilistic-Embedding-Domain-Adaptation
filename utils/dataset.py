import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class UnpaddedDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: array of [sequence, label, domain_id]
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx][0], self.data[idx][1]

        seq = torch.tensor(seq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return seq, label


class ConcatDataset(Dataset):
    def __init__(self, src_dataset, tgt_dataset):
        assert len(src_dataset) == len(tgt_dataset)
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset

    def __len__(self):
        return len(self.src_dataset)

    def __getitem__(self, idx):
        src_seq, src_label = self.src_dataset[idx]
        tgt_seq, tgt_label = self.tgt_dataset[idx]
        return (src_seq, src_label), (tgt_seq, tgt_label)


def split_dataset_for_da(src_dataset, tgt_dataset, random_seed):
    src_size = len(src_dataset)
    tgt_size = len(tgt_dataset)
    length = min(src_size, tgt_size)

    train_size = int(0.6 * length)
    val_size = int(0.2 * length)
    test_size = length - train_size - val_size

    np.random.seed(random_seed)

    src_indices = np.random.permutation(src_size)[:length]
    tgt_indices = np.random.permutation(tgt_size)[:length]

    src_train = Subset(src_dataset, src_indices[:train_size])
    src_val = Subset(src_dataset, src_indices[train_size:train_size+val_size])
    src_test = Subset(src_dataset, src_indices[train_size+val_size:])

    tgt_train = Subset(tgt_dataset, tgt_indices[:train_size])
    tgt_val = Subset(tgt_dataset, tgt_indices[train_size:train_size+val_size])
    tgt_test = Subset(tgt_dataset, tgt_indices[train_size+val_size:])

    return [[src_train, src_val, src_test], [tgt_train, tgt_val, tgt_test]]
