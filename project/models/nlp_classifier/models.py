from torch import nn

from transformers import BertForSequenceClassification

class BertPretrainedClassification(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=config["output_size"], 
            hidden_size=config["hidden_size"]
        )

        # Freeze BERT layers, keeping classification layers unfreezed
        for param in self.model.bert.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(**x)
