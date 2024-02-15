import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io 
import os
from joblib import Parallel, delayed
from LibMTL.weighting.abstract_weighting import AbsWeighting

import matlab.engine

class ConfMax(AbsWeighting):

    def __init__(self):
        super(ConfMax, self).__init__()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd('/home/pasch/repos/LibMTL')
    
    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    ind = []
                    beg = 0
                    for g in grad:
                        cnt = g.view(-1).size()[0]
                        beg += cnt
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
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
        return grads
        
    def backward(self, losses, **kwargs):
        batch_weight = np.ones(len(losses))
        if self.rep_grad:
            raise ValueError('No support method PCGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        # pc_grads = grads.clone()
        
        # Limited to the case where self.task_num == 2
        if self.task_num != 2 :
            raise NotImplementedError("ConfMax only works for two tasks as of now.")
        
        # Sample task gradient to modify
        tn_mod = np.random.choice(range(self.task_num))
        tn_stat = 1 - tn_mod

        np_grads = grads.cpu().numpy()
        if self.grad_dim < 30000:
            if torch.dot(grads[tn_mod], grads[tn_stat]) > 0:

                x, exitflag = self.eng.solve_socp(np_grads[tn_mod].reshape(-1,1).astype(float),
                                                np_grads[tn_stat].reshape(-1,1).astype(float),
                                                self.grad_dim, kwargs['ConfMax_retain'], nargout=2, stdout=io.StringIO(), stderr=io.StringIO())
                if int(exitflag) == 1:
                    new_grads = [grads[tn_stat]]
                    new_grads.insert(tn_mod, 
                                    torch.tensor(np.asarray(x).copy().astype(np.float32)).reshape(-1,).to(device=grads.device))
                    self._reset_grad(torch.stack(new_grads))
        else:
            # Per module gradient, for conv modules per channel gradients
            new_grad = np.zeros_like(np_grads[tn_mod])
            beg = 0
            for (name, param), s in zip(self.encoder.named_parameters(),
                                         self.grad_index):
                # print(f'ConfMax {name}, size: {param.size()}')
                if torch.dot(grads[tn_mod, beg:beg+s], grads[tn_stat, beg:beg+s]) > 0:
                    # If the module parameter length is under 30000, just do it as one problem
                    if s < 30000:
                        x, exitflag = self.eng.solve_socp(np_grads[tn_mod, beg:beg+s].reshape(-1,1).astype(float),
                                                        np_grads[tn_stat, beg:beg+s].reshape(-1,1).astype(float),
                                                        s, kwargs['ConfMax_retain'], nargout=2, stdout=io.StringIO(), stderr=io.StringIO())
                        if int(exitflag) == 1:
                            new_grad[beg:beg+s] = np.asarray(x).copy().reshape(-1,)
                        else:
                            new_grad[beg:beg+s] = np_grads[tn_mod, beg:beg+s]
                    else:
                        # Split it up across channel (or, first dimension)
                        n_channels = param.size()[0]
                        part_size = np.prod(param.size()[1:]).astype(int)
                        assert part_size < 30000
                        out = io.StringIO()
                        np.savetxt('/home/pasch/repos/LibMTL/np_grads_mod.txt', np_grads[tn_mod, beg:beg+s].reshape(-1,1).astype(float))
                        np.savetxt('/home/pasch/repos/LibMTL/np_grads_stat.txt', np_grads[tn_stat, beg:beg+s].reshape(-1,1).astype(float))
                        self.eng.eval("grads_mod = readmatrix('np_grads_mod.txt')", nargout=0, stdout=out)
                        self.eng.eval("grads_stat = readmatrix('np_grads_stat.txt')", nargout=0, stdout=out)
                        x, exitflag = self.eng.eval(f'solve_socp_parallel(grads_mod, grads_stat, {n_channels}, {part_size}, {kwargs["ConfMax_retain"]})', nargout=2, stdout=out, stderr=out)
                        # x, exitflag = self.eng.solve_socp_parallel(np_grads[tn_mod, beg:beg+s].reshape(-1,1).astype(float),
                                                        # np_grads[tn_stat, beg:beg+s].reshape(-1,1).astype(float),
                                                        # n_channels, part_size, kwargs['ConfMax_retain'], nargout=2, stdout=io.StringIO(), stderr=io.StringIO())
                        if np.asarray(exitflag).all():
                            new_grad[beg:beg+s] = np.asarray(x).copy().reshape(-1,)
                        else:
                            new_grad[beg:beg+s] = np_grads[tn_mod, beg:beg+s]
                            # for i in range(n_channels):
                            
                            # print(f"{i}, part size {part_size}")
                            # if torch.dot(grads[tn_mod, beg+i*part_size:beg+(i+1)*part_size],
                            #              grads[tn_mod, beg+i*part_size:beg+(i+1)*part_size]) > 0:
                            #     x, exitflag = self.eng.solve_socp(np_grads[tn_mod, beg+i*part_size:beg+(i+1)*part_size].reshape(-1,1).astype(float),
                            #                                     np_grads[tn_stat, beg+i*part_size:beg+(i+1)*part_size].reshape(-1,1).astype(float),
                            #                                     part_size, kwargs['ConfMax_retain'], nargout=2, stdout=io.StringIO(), stderr=io.StringIO())
                            #     if int(exitflag) == 1:
                            #         new_grad[beg+i*part_size:beg+(i+1)*part_size] = np.asarray(x).reshape(-1,)
                            #     else:
                            #         new_grad[beg+i*part_size:beg+(i+1)*part_size] = np_grads[tn_mod,beg+i*part_size:beg+(i+1)*part_size]
                            # else:
                            #     new_grad[beg+i*part_size:beg+(i+1)*part_size] = np_grads[tn_mod,beg+i*part_size:beg+(i+1)*part_size]
                beg += s
            new_grads = [grads[tn_stat]]
            new_grads.insert(tn_mod, 
                             torch.tensor(new_grad.copy().astype(np.float32)).reshape(-1,).to(device=grads.device))
            self._reset_grad(torch.stack(new_grads))
        return torch.ones(self.task_num)
    
    def __channelwise_opt(self, np_grads, new_grad, part_size, i, tn_mod, tn_stat, kwargs, beg):
        if np.dot(np_grads[tn_mod, beg+i*part_size:beg+(i+1)*part_size],
                                        np_grads[tn_mod, beg+i*part_size:beg+(i+1)*part_size]) > 0:
            x, exitflag = self.eng[i%4].solve_socp(np_grads[tn_mod, beg+i*part_size:beg+(i+1)*part_size].reshape(-1,1).astype(float),
                                                            np_grads[tn_stat, beg+i*part_size:beg+(i+1)*part_size].reshape(-1,1).astype(float),
                                                            part_size, kwargs['ConfMax_retain'], nargout=2, stdout=io.StringIO(), stderr=io.StringIO())
            if int(exitflag) == 1:
                new_grad[beg+i*part_size:beg+(i+1)*part_size] = np.asarray(x).reshape(-1,)
            else:
                new_grad[beg+i*part_size:beg+(i+1)*part_size] = np_grads[tn_mod,beg+i*part_size:beg+(i+1)*part_size]
        else:
            new_grad[beg+i*part_size:beg+(i+1)*part_size] = np_grads[tn_mod,beg+i*part_size:beg+(i+1)*part_size]
