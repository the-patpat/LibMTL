# Copied from 
# https://github.com/yobibyte/unitary-scalarization-dmtl/blob/main/supervised_experiments/models/multi_lenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

class MultiLeNetBackbone(nn.Module):
    def __init__(self, no_dropout=False):
        super(MultiLeNetBackbone, self).__init__()
        self._conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self._conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self._conv2_drop = nn.Dropout2d()
        self._fc = nn.Linear(320, 50)
        self.no_dropout = no_dropout

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1, channel_size, 1, 1) * 0.5).to(x.device))
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x, mask=None):
        x = F.relu(F.max_pool2d(self._conv1(x), 2))
        x = self._conv2(x)
        mask = self.dropout2dwithmask(x, mask)
        if self.training and not self.no_dropout:
            x = x * mask
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self._fc(x))
        return x


class MultiLeNetHead(nn.Module):
    def __init__(self, no_dropout=False, n_classes=10):
        super(MultiLeNetHead, self).__init__()
        self._fc1, self._fc2 = nn.Linear(50, 50), nn.Linear(50, n_classes)
        self.no_dropout = no_dropout

    def forward(self, x, mask=None):
        x = F.relu(self._fc1(x))
        if mask is None:
            mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training and not self.no_dropout:
            x = x * mask
        x = self._fc2(x)
        return x