import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from dataset import MultiMNIST

from LibMTL import Trainer
from LibMTL.metrics import AccMetric
from LibMTL.loss import CELoss
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
import wandb
import multi_lenet
import numpy as np
import random

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_bs', default=100, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=100, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dropout', action='store_true', help='use dropout for multilenet')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(params):

    run = wandb.init('libmtl_gradient_trajectory')
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # Instantiate dataloaders
    g = torch.Generator()
    g.manual_seed(params.seed)
    t = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])
    trainloader = torch.utils.data.DataLoader(
        MultiMNIST(root=params.dataset_path, split='train', 
                             transform=t),
        batch_size=params.train_bs, 
        shuffle=True, 
        num_workers=0, 
        worker_init_fn=seed_worker,
        generator=g
    )
    valloader = torch.utils.data.DataLoader(
        MultiMNIST(root=params.dataset_path, split='val',
                           transform=t),
        batch_size=params.train_bs, 
        shuffle=True, 
        num_workers=0, 
        worker_init_fn=seed_worker,
        generator=g
    )
    testloader = torch.utils.data.DataLoader(
        MultiMNIST(root=params.dataset_path, split='test',
                           transform=t),
        batch_size=params.train_bs, 
        shuffle=True, 
        num_workers=0, 
        worker_init_fn=seed_worker,
        generator=g
    )
    
    batch = next(iter(testloader))
    
    # define tasks 
    task_dict = {
        'L' : {'metrics': ['classAcc'], 
               'metrics_fn' : AccMetric(), 
               'loss_fn' : CELoss(), 
               'weight' : [1]
        }, 
        'R' : {'metrics': ['classAcc'], 
               'metrics_fn' : AccMetric(), 
               'loss_fn' : CELoss(), 
               'weight' : [1]
        } 
    }
    
    # Define encoder and decoders
    encoder_class = lambda : multi_lenet.MultiLeNetBackbone(params.dropout).to(f'cuda:{params.gpu_id}')
    decoders = {task : multi_lenet.MultiLeNetHead(params.dropout).to(f'cuda:{params.gpu_id}')
                for task in task_dict.keys()}

    class MultiMNISTTrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param,
                     scheduler_param, wandb_run, **kwargs):
            super(MultiMNISTTrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting_method.__dict__[weighting], 
                                            architecture=architecture_method.__dict__[architecture], 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            wandb_run=wandb_run,
                                            **kwargs)
    MNISTModel = MultiMNISTTrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          wandb_run=run,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    if params.mode == 'train':
        MNISTModel.train(trainloader, valloader, params.epochs)
    elif params.mode == 'test':
        MNISTModel.test(testloader)
    else:
        raise ValueError

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)

