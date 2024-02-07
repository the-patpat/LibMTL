import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cvxopt import matrix, spmatrix, sparse, solvers
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from LibMTL.weighting.abstract_weighting import AbsWeighting

import matlab.engine

class ConfMax(AbsWeighting):

    def __init__(self):
        super(ConfMax, self).__init__()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd('/home/pasch/repos/LibMTL')
    
    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        inds = []
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
                inds.append((beg, end))
            count += 1
        return grad, inds

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            inds = []
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn], ind = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    ind = []
                    beg = 0
                    for g in grad:
                        cnt = g.view(-1).size()[0]
                        ind.append((beg, (beg+cnt-1)))
                        beg += cnt
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                inds.append(ind)
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads, inds
        
    def backward(self, losses, **kwargs):
        batch_weight = np.ones(len(losses))
        if self.rep_grad:
            raise ValueError('No support method PCGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads, inds = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        # pc_grads = grads.clone()
        
        # Limited to the case where self.task_num == 2
        if self.task_num != 2 :
            raise NotImplementedError("ConfMax only works for two tasks as of now.")
        
        # Sample task gradient to modify
        tn_mod = np.random.choice(range(self.task_num))
        tn_stat = 1 - tn_mod

        if torch.dot(grads[tn_mod], grads[tn_stat]) > 0:
            np_grads = grads.cpu().numpy()

            x, exitflag = self.eng.solve_socp(np_grads[tn_mod].reshape(-1,1).astype(float),
                                            np_grads[tn_stat].reshape(-1,1).astype(float),
                                            self.grad_dim, nargout=2)
            if int(exitflag) == 1:
                new_grads = [grads[tn_stat]]
                new_grads.insert(tn_mod, 
                                torch.tensor(np.asarray(x).copy()).reshape(-1,).to(device=grads.device))
                self._reset_grad(torch.stack(new_grads))
        return torch.ones(self.task_num) 