import numpy as np
import torch
from LibMTL.weighting import AbsWeighting

class AbsScheduledWeighting(AbsWeighting):
    def __init__(self):
        super(AbsScheduledWeighting, self).__init__()
    
    def _backward(self, losses, **kwargs):
        pass
    
    def backward(self, losses, **kwargs):
        if self.epoch > 10:
            # Do EW in the beginning
            loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
            loss.backward()
            return np.ones(self.task_num)
        else:
            return self._backward(losses, **kwargs)
