from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision import models


class EfficientNetPretrained(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b1')

        for param in self.model.parameters():
            param.requires_grad = True

        self.model._fc = nn.Sequential(
            nn.Linear(self.model._fc.in_features, config["lin1_size"]),
            nn.BatchNorm1d(config["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(config["lin1_size"], config["lin2_size"]),
            nn.BatchNorm1d(config["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(config["lin2_size"], config["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


class ResnetPretrained(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        for name, param in self.model.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, config["lin1_size"]),
            nn.BatchNorm1d(config["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(config["lin1_size"], config["lin2_size"]),
            nn.BatchNorm1d(config["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["lin2_size"], config["output_size"]),
        )

    def forward(self, x):
        return self.model(x)


class ResnextPretrained(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = models.resnext101_32x8d(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, config["lin1_size"]),
            nn.BatchNorm1d(config["lin1_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(config["lin1_size"], config["lin2_size"]),
            nn.BatchNorm1d(config["lin2_size"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["lin2_size"], config["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
