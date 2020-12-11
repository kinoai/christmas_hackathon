from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    """
    NLP DataModule.
    """
    def __init__(self, hparams):
        super().__init__()

        # hparams["data_dir"] is always automatically set to "path_to_project/data/"
        self.data_dir = hparams["data_dir"]

        self.batch_size = hparams.get("batch_size") or 64
        self.train_val_split_ratio = hparams.get("train_val_split_ratio") or 0.9
        self.num_workers = hparams.get("num_workers") or 1
        self.pin_memory = hparams.get("pin_memory") or False

        self.transforms = transforms.ToTensor()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
