import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(torch.nn.Module):
    def __init__(self, in_channel, num_classes=2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, num_classes)
        )
    def forward(self, input):
      return self.net(input)    

class WeightMLP(torch.nn.Module):
    def __init__(self, in_channel, num_classes=2):
        super(WeightMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, num_classes)
        )
    def forward(self, input):
      return self.net(input)  