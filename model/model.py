import torch
import torch.nn as nn
from resnet import *

# Define the CLS model
class CLSModel(nn.Module):
    def __init__(self,pretrain, classes):
        super(CLSModel, self).__init__()
        if pretrain == "resnet50":
            self.resnet50 = resnet50_quantizable(pretrained=True)
        if pretrain == "resnet152":
            self.resnet50 = resnet50_quantizable(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, classes)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 120)
        self.BN1 = nn.BatchNorm1d(1024)
        self.BN2 = nn.BatchNorm1d(512)
        self.BN3 = nn.BatchNorm1d(256)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet50(x)
        # x = self.BN1(x)
        # x = self.dropout1(x)
        # x = self.fc1(x)
        # x = nn.functional.relu(x)
        # x = self.BN2(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # x = nn.functional.relu(x)
        # x = self.BN3(x)
        # x = self.fc3(x)
        return x


