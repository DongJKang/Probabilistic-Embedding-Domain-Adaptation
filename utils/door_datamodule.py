import lightning as L
from utils.dataset import *

class DoorDataModule(L.LightningDataModule):
    def __init__(self, src_str, tgt_str, batch_size, seed):
        super().__init__()
        self.src_str = src_str
        self.tgt_str = tgt_str
        self.batch_size = batch_size
        self.seed = seed
        self.dataset = self.load_dataset()

    def train_dataloader(self):
        return DataLoader(
            ConcatDataset(self.dataset['src_train'], self.dataset['tgt_train']),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=3,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ConcatDataset(self.dataset['src_val'], self.dataset['tgt_val']),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers = 3,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            ConcatDataset(self.dataset['src_test'], self.dataset['tgt_test']),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=3,
            persistent_workers=True,
        )

    def load_dataset(self):
        data = np.load('./data/door_dataset_v5.npy', allow_pickle=True)

        domain_map = {
            'Lisa_2c': 0,
            'Ryan_2c': 1,
            'Lisa_4c': 2,
            'Ryan_4c': 3,
            'C2D5': 4,
            'C2D8': 5,
        }

        src_idx = domain_map[self.src_str]
        tgt_idx = domain_map[self.tgt_str]

        src_data = data[data[:, 2] == src_idx]
        tgt_data = data[data[:, 2] == tgt_idx]

        src_dataset = UnpaddedDataset(src_data)
        tgt_dataset = UnpaddedDataset(tgt_data)

        dataset = split_dataset_for_da(src_dataset, tgt_dataset, self.seed)

        return {
            'src_train': dataset[0][0],
            'src_val': dataset[0][1],
            'src_test': dataset[0][2],
            'tgt_train': dataset[1][0],
            'tgt_val': dataset[1][1],
            'tgt_test': dataset[1][2]
        }