import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from dataset import CityScape

from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from utils import SegLoss, SegMetric, DepthLoss, DepthMetric
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from aspp import DeepLabHead
import wandb
import numpy as np
import random
import re

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_bs', default=100, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=100, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(params):

    run = wandb.init('libmtl_gradient_trajectory')
    kwargs, optim_param, scheduler_param = prepare_args(params)
    wandb.config.update(
        {
            'cli' : params.__dict__,
            'opt' : optim_param,
            'scheduler' : scheduler_param,
            'dataset' : 'cityscapes'
        }
    )

    # Instantiate dataloaders
    g = torch.Generator()
    g.manual_seed(params.seed)

    trainloader = torch.utils.data.DataLoader(
        CityScape(root=params.dataset_path, mode='train', 
                             augmentation=params.aug),
        batch_size=params.train_bs, 
        shuffle=True, 
        num_workers=0, 
        worker_init_fn=seed_worker,
        generator=g
    )
    valloader = torch.utils.data.DataLoader(
        CityScape(root=params.dataset_path, mode='val',
                           augmentation=params.aug),
        batch_size=params.train_bs, 
        shuffle=True, 
        num_workers=0, 
        worker_init_fn=seed_worker,
        generator=g
    )

    # define tasks 
    task_dict = {'segmentation': {'metrics':['mIoU', 'pixAcc'], 
                              'metrics_fn': SegMetric(7),
                              'loss_fn': SegLoss(),
                              'weight': [1, 1]}, 
                 'depth': {'metrics':['abs_err', 'rel_err'], 
                           'metrics_fn': DepthMetric(),
                           'loss_fn': DepthLoss(),
                           'weight': [0, 0]}
    }

    # Define encoder and decoders
    def encoder_class(): 
        return resnet_dilated('resnet18', pretrained=True)
    num_out_channels = {'segmentation': 7, 'depth': 1}
    decoders = nn.ModuleDict({task: DeepLabHead(512, 
                                                num_out_channels[task]) for task in list(task_dict.keys())})

    # Define custom trainer. Does not change much. Can change e.g. when predictions etc need to be processed.
    class CityscapesTrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param,
                     scheduler_param, wandb_run, **kwargs):
            super(CityscapesTrainer, self).__init__(task_dict=task_dict, 
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
        def process_preds(self, preds):
            for task in self.task_name:
                preds[task] = F.interpolate(preds[task], (128, 256), mode='bilinear', align_corners=True)
            return preds
    
    # Instantiate trainer
    CityscapesModel = CityscapesTrainer(task_dict=task_dict, 
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
        CityscapesModel.train(trainloader, None, params.epochs,
                         val_dataloaders=valloader)
    else:
        raise ValueError

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)

