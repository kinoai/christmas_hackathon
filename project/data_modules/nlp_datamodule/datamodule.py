import os
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from transformers import BertTokenizer

from data_modules.nlp_datamodule.datasets import CsvDataset

DATA_SUBDIR = 'nlp'

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

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Model parameter
        MAX_SEQ_LEN = 512
        # PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        # UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

        train_file = os.path.join(self.data_dir, DATA_SUBDIR, 'train.csv')
        trainset = CsvDataset(train_file, tokenizer, MAX_SEQ_LEN)

        train_length = int(len(trainset) * self.train_val_split_ratio)
        val_length = len(trainset) - train_length
        train_val_split = [train_length, val_length]

        self.data_train, self.data_val = random_split(trainset, train_val_split)
        self.data_test = None

        # Fields

        # label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        # text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
        #                 fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
        # fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]

        # # TabularDataset

        # train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
        #                                         test='test.csv', format='CSV', fields=fields, skip_header=True)

        # Iterators

        # train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
        #                             device=device, train=True, sort=True, sort_within_batch=True)
        # valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
        #                             device=device, train=True, sort=True, sort_within_batch=True)
        # test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
