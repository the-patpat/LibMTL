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

        np_grads = grads.cpu().numpy()

        x, exitflag = self.eng.solve_socp(np_grads[tn_mod].reshape(-1,1).astype(float),
                                          np_grads[tn_stat].reshape(-1,1).astype(float),
                                          self.grad_dim, nargout=2)
        if int(exitflag) == 1:
            new_grads = [grads[tn_stat]]
            new_grads.insert(tn_mod, 
                             torch.tensor(np.asarray(x).copy()).reshape(-1,).to(device=grads.device))
            self._backward_new_grads(torch.ones(self.task_num),
                                     grads=torch.stack(new_grads))

        # sol = minimize(
        #     lambda x: np.dot(np_grads[tn_stat],x),
        #     np.zeros_like(np_grads[tn_stat]),
        #     constraints=[
        #         LinearConstraint(np.vstack((-np_grads[tn_stat], np_grads[tn_mod])), ub=np.asarray([0,0])),
        #         NonlinearConstraint(np.linalg.norm, lb=-np.inf, ub=np.linalg.norm(np_grads[tn_stat]))
        #     ], 
        #     # method='trust-constr' 
        # )
        # if sol.x is not None:
        #     new_grads = [grads[tn_stat]]
        #     new_grads.insert(tn_mod,
        #                      torch.Tensor(sol.x).to(device=grads.device))
        #     self._backward_new_grads(torch.ones(self.task_num),
        #                              grads=torch.cat(new_grads))
        # # Formulate problem
        # c = matrix(np.asarray(grads[tn_stat].cpu(), dtype=float))
        # G = matrix([
        #     matrix(np.asarray(-grads[tn_mod].cpu(), dtype=float)).T,
        #     matrix(np.asarray(grads[tn_stat].cpu(), dtype=float)).T,
        #     matrix(np.zeros(self.grad_dim, dtype=float)).T,
        #     spmatrix(1.0, range(self.grad_dim), range(self.grad_dim))
        # ])
        # h = matrix([matrix([0.0, 0.0, float(grads[tn_mod].cpu().norm())]),
        #             matrix(np.zeros(self.grad_dim, dtype=float))])
        # dims = {'l' : 2, 'q' : [self.grad_dim+1], 's' : []}
        # sol = solvers.conelp(c, G, h, dims)

        # if sol['x'] is not None:
        #     new_grads = [grads[tn_stat]]
        #     new_grads.insert(tn_mod, 
        #                      torch.Tensor(np.asarray(sol['x']).T).to(grads.device))
        #     self._backward_new_grads(torch.ones(self.task_num), 
        #                              grads=torch.cat(new_grads))
        else:
            loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
            loss.backward()

        del x, exitflag


        # for tn_i in range(self.task_num):
        #     task_index = list(range(self.task_num))
        #     random.shuffle(task_index)
        #     for tn_j in task_index:
        #         g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
        #         if g_ij < 0:
        #             pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2)+1e-8)
        #             batch_weight[tn_j] -= (g_ij/(grads[tn_j].norm().pow(2)+1e-8)).item()
        # new_grads = pc_grads.sum(0)
        # self._reset_grad(new_grads)
        return torch.ones(self.task_num) 