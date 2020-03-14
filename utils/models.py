import torch.nn as nn

from torchvision import models


class CLSTM(nn.Module):
    def __init__(self, freeze, hidden_state):
        super(CLSTM, self).__init__()

        self.conv_model = models.resnet18(True)

        if freeze:
            for param in self.conv_model.parameters():
                param.require_grad = False

        self.lstm = nn.LSTM(self.conv_model.fc.in_features, hidden_state)

        self.affine = nn.Linear(hidden_state, 4)

    def forward(self, x):
        CNN_codes = self.conv_model(x)
        lstm_out, _ = self.lstm(CNN_codes.view(len(x), 1, -1))
        out = self.affine(lstm_out.view(len(x), -1))
        return out


resnet18 = models.resnet18(True)
for param in resnet18.parameters():
    param.require_grad = False

resnet18.fc = nn.Linear(resnet18.fc.in_features, 4)
