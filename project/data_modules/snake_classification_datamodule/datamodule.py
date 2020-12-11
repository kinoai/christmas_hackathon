import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

torch.manual_seed(666)


class DataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()

        self.data_dir = hparams["data_dir"] + "/snake_dataset/train"
        self.batch_size = hparams.get("batch_size") or 64
        self.train_val_split_ratio = hparams.get("train_val_split_ratio") or 0.9
        self.num_workers = hparams.get("num_workers") or 1
        self.pin_memory = hparams.get("pin_memory") or False

        img_augmentation_transformations = [
            # transforms.RandomAffine((-15, 15), translate=(0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation((-15, 15)),
            transforms.ColorJitter(hue=.05, saturation=.05),
        ]

        self.transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.RandomApply(img_augmentation_transformations, 0.5),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
         ])

        self.data_train = None
        self.data_val = None

    def setup(self, stage=None):
        trainset = ImageFolder(self.data_dir, transform=self.transforms)
        train_length = int(len(trainset) * self.train_val_split_ratio)
        val_length = len(trainset) - train_length
        train_val_split = [train_length, val_length]
        
        self.data_train, self.data_val = random_split(trainset, train_val_split)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
