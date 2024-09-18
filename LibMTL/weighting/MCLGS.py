from LibMTL.weighting.abstract_weighting import AbsWeighting
import torch

class MCLGS(AbsWeighting):
    """ Multi-Task Curriculum Learning based on Gradient Similarity
    https://bmvc2022.mpi-inf.mpg.de/0705.pdf
    """
    def __init__(self):
        super(MCLGS, self).__init__()

    def weight_fun(self, g1,g2,t,a0,ra):
        # Linear decay as in paper
        da = a0/(ra*self.epochs)
        p = max(a0 - t*da, 0)
        return torch.tanh(torch.dot(g1, g2)/(g1.norm()*g2.norm()) * p) + 1

    def backward(self, losses, **kwargs):
        """_summary_

        Args:
            losses (list): List of losses for each task
        """

        a0, ra = kwargs['a0'], kwargs['ra']
        # Get the per-task gradients
        self._compute_grad_dim()
        g = self._compute_grad(losses, mode='autograd') # List of gradients wrt shared parameters

        # Weight matrix to store w_ij(t)
        w = torch.zeros((self.task_num, self.task_num), 
                        device=g.device)

        # New gradient to store the sum in
        for i in range(self.task_num-1):
            for j in range(i+1, self.task_num):
                w[i,j] = self.weight_fun(g[i], g[j],
                                          self.epoch, a0, ra)

        # Weight vector
        w_vector = w.sum(dim=0) + w.sum(dim=1)
        w_vector /= (self.task_num*(self.task_num-1))
        loss = (w_vector*losses).sum()
        loss.backward()
        return w_vector.detach().cpu().numpy()
