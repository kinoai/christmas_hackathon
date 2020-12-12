from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import pytorch_lightning as pl
import torch

# custom models
from models.nlp_classifier.models import *


class LitModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()

        if hparams:
            self.save_hyperparameters(hparams)

        self.model = BertPretrainedClassification(config=self.hparams)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = torch.sigmoid(self.model(x).logits)
        loss = F.binary_cross_entropy(logits, y)

        # training metrics
        preds = torch.where(logits > 0.5, 1, 0).cpu()
        y = y.cpu()
        acc = accuracy_score(preds, y)
        p = precision_score(preds, y, average="micro")
        r = recall_score(preds, y, average="micro")
        f1 = f1_score(preds, y, average="micro")

        self.log('train_f1_score', f1, on_epoch=True, on_step=False, logger=True)
        self.log('train_precision', p, on_epoch=True, on_step=False, logger=True)
        self.log('train_recall', r, on_epoch=True, on_step=False, logger=True)
        self.log('train_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('train_acc', acc, on_epoch=True, on_step=False, logger=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = torch.sigmoid(self.model(x).logits)
        loss = F.binary_cross_entropy(logits, y)

        # training metrics
        preds = torch.where(logits > 0.5, 1, 0).cpu()
        y = y.cpu()
        acc = accuracy_score(preds, y)
        p = precision_score(preds, y, average="micro")
        r = recall_score(preds, y, average="micro")
        f1 = f1_score(preds, y, average="micro")

        self.log('val_f1_score', f1, on_epoch=True, on_step=False, logger=True)
        self.log('val_precision', p, on_epoch=True, on_step=False, logger=True)
        self.log('val_recall', r, on_epoch=True, on_step=False, logger=True)
        self.log('val_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('val_acc', acc, on_epoch=True, on_step=False, logger=True)

        # we can return here anything and then read it in some callback
        return preds, y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams["weight_decay"])
