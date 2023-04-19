import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import copy
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def init_toy_optimizer(model, cfg):
    optim_cfg = cfg["OPTIMIZER"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["LR"])
    scheduler = MultiStepLR(optimizer, optim_cfg["DECAY_STEPS"], gamma=0.1)
    return optimizer, scheduler

def init_optimizer(model, cfg):
    optim_cfg = cfg["OPTIMIZER"]
    b_params = [param for name, param in model.named_parameters() if 'backbone' in name]
    s_params = [param for name, param in model.named_parameters() if 'backbone' not in name]
    optimizer = torch.optim.AdamW([
    {'params': b_params, 'lr': optim_cfg["B_LR"]},
    {'params': s_params, 'lr': optim_cfg["A_LR"]}
    ])
    scheduler = MultiStepLR(optimizer, optim_cfg["DECAY_STEPS"], gamma=0.1)
    return optimizer, scheduler

    
    
    
    

def adapt_lr(optimizer, global_iter, base_lr, max_iters): 
    cur_iters = global_iter
    warmup_iters = 3000
    warmup_ratio = 1e-06
    # print("self.max_iters", self.max_iters)
    # print('base_lr', self.base_lr)
    # print('all_epochs', self.total_epoch_nums)
    lr_ = base_lr * (1.0 - (cur_iters - 1) / max_iters) ** 1.0
    
    for param_group in optimizer.param_groups:
        # print("past learning rate", param_group["lr"])
        param_group['lr'] = lr_
    return lr_
        # print("current learning rate", param_group["lr"])