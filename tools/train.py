import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import os
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn import functional as F

from depth_c2rp.models.heads import FaPNHead
from depth_c2rp.models.backbones import ResNet
from depth_c2rp.optimizers import get_optimizer
from depth_c2rp.build import build_model
from depth_c2rp.losses import get_loss
from depth_c2rp.utils.utils import save_model, load_model, exists_or_mkdir, visualize_training_loss
from depth_c2rp.configs.config import update_config


def main(cfg):
    """
        This code is designed for the whole training. 
    """
    # expect cfg is a dict
    assert type(cfg) == dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    
    # Build DataLoader
    
    
    # Build Model
    model_cfg = cfg["MODEL"]
    model = build_model(model_cfg["BACKBONE"], model_cfg["HEAD"], model_cfg["MODEL_CLASSES"], model_cfg["IN_CHANNELS"])
    model.init_pretrained(model_cfg["PRETRAINED"])
    
    # Build Loss Function
    loss_cfg = cfg["LOSS"]
    loss_fn = get_loss(loss_cfg["NAME"], int(loss_cfg["IGNORE_LABELS"]))
    
    # Build Optimizer
    optim_cfg = cfg["OPTIMIZER"]
    optimizer = get_optimizer(model, optim_cfg["NAME"], optim_cfg["LR"], optim_cfg["WEIGHT_DECAY"])
    
    # Build Recording and Saving Path
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    checkpoint_path = os.path.join(save_path, "CHECKPOINT")
    tb_path = os.path.join(save_path, "TB")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    writer = SummaryWriter(tb_path)
    
    # Load Checkpoint / Resume Training
    start_epoch = 0
    if cfg["RESUME"]:
        print("Resume Model Checkpoint")
        checkpoint_paths = os.listdir(checkpoint_path)
        checkpoint_paths.sort() 
        this_ckpt_path = os.path.join(checkpoint_path, checkpoint_paths[-1])
        print('this_ckpt_path', this_ckpt_path)
        model, optimizer, start_epoch = load_model(model, this_ckpt_path, optim_cfg["LR"], optimizer)
        
    
    # Training 
    model.train()
    model = model.to(device)
    train_cfg = cfg["TRAIN"]
    for epoch in tqdm(range(start_epoch + 1, train_cfg["EPOCHS"] + 1)):
        start_time = time.time()
        x = torch.randn(1, 3, 720, 1280).to(device)
        label = torch.ones((1,720, 1280), dtype=torch.long).to(device)
        middle_time = time.time() 
        
        out = model(x)
        end_time = time.time()
        
        print("Image Construction Time", middle_time - start_time)
        print("Model Time", end_time - middle_time)
        
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # Visualization
    
    
        # Save Output and Checkpoint
        with torch.no_grad():
            # Save Checkpoint
            save_model(os.path.join(checkpoint_path, "model_{}.pth".format(str(epoch).zfill(3))), epoch, model, optimizer)


if __name__ == "__main__":
    cfg, args = update_config()
    main(cfg)

















