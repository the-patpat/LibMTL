import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CelebAHead(nn.Module):
    def __init__(self, no_dropout=False, n_classes=10):
        super(CelebAHead, self).__init__()
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