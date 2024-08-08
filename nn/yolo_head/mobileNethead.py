import torch.nn as nn


class MobileNetV4_head(nn.Module):
    def __init__(self, c1,c2):
        super(MobileNetV4_head, self).__init__()
        hidden_channels=1280
        self.classifier = nn.Linear(hidden_channels, c2)
    def forward(self, x):
        x = self.classifier(x)
        return x if self.training else x.softmax(1)