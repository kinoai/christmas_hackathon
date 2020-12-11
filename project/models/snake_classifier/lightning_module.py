from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import pytorch_lightning as pl
import torch

# custom models
from models.snake_classifier.models import *


class LitModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        if hparams:
            self.save_hyperparameters(hparams)

        self.model = ResnetPretrained(config=self.hparams)

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        preds, y = preds.cpu(), y.cpu()
        acc = accuracy_score(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        preds, y = preds.cpu(), y.cpu()
        acc = accuracy_score(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return preds, y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams["weight_decay"])
