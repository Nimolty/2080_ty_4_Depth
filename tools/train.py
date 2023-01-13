import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
from depth_c2rp.optimizers import get_optimizer, adapt_lr
from depth_c2rp.build import build_model
from depth_c2rp.losses import get_loss, Calculate_Loss
from depth_c2rp.utils.utils import save_model, load_model, exists_or_mkdir, visualize_training_loss, find_seq_data_in_dir
from depth_c2rp.configs.config import update_config
from depth_c2rp.datasets.datasets import Depth_dataset
from inference import network_inference


def main(cfg):
    """
        This code is designed for the whole training. 
    """
    # expect cfg is a dict
    assert type(cfg) == dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    
    # Build DataLoader
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    training_data_dir = dataset_cfg["TRAINING_ROOT"]
    training_data = find_seq_data_in_dir(training_data_dir)
    training_dataset = Depth_dataset(training_data, dataset_cfg["MANIPULATOR"], dataset_cfg["KEYPOINT_NAMES"], dataset_cfg["JOINT_NAMES"], \
                                     dataset_cfg["INPUT_RESOLUTION"], mask_dict=dataset_cfg["MASK_DICT"])
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=train_cfg["BATCH_SIZE"], shuffle=True, \
                                               num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True)
    
    # Build Model
    model_cfg = cfg["MODEL"]
    model = build_model(model_cfg["BACKBONE"], model_cfg["HEAD"], model_cfg["MODEL_CLASSES"], model_cfg["IN_CHANNELS"], \
                        dataset_cfg["NUM_JOINTS"], dataset_cfg["OUTPUT_RESOLUTION"][0], dataset_cfg["OUTPUT_RESOLUTION"][1])
    model.init_pretrained(model_cfg["PRETRAINED"])
    
    # Build Loss Function
    loss_cfg = cfg["LOSS"]
    loss_fn_names = {"masks" : loss_cfg["NAME"]}
    loss_fn = Calculate_Loss(loss_fn_names,weights=loss_cfg["WEIGHTS"])
    
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
    device_ids = [6, 7] 	
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    device = torch.device('cuda:{}'.format(device_ids[0]) if torch.cuda.is_available() else 'cpu' )
    model = model.to(device)    
    max_iters = len(training_loader) * train_cfg["EPOCHS"]
    print("len_dataloader", len(training_loader))
    
    for epoch in tqdm(range(start_epoch + 1, train_cfg["EPOCHS"] + 1)):
        model.train()
        for batch_idx, batch in tqdm(enumerate(training_loader)):
            adapt_lr(optimizer, epoch, batch_idx, len(training_loader), optim_cfg["LR"], max_iters)
            start_time = time.time()
#            if batch_idx > 1:
#                break
            
            next_img = batch["next_frame_img_as_input"].to(device)
            next_xy_wrt_cam, next_uv, next_simdepth, next_normals = batch["next_frame_xy_wrt_cam"].to(device), batch["next_frame_uv"].to(device), \
            batch["next_frame_simdepth_as_input"].to(device), batch["next_normals_crop"].to(device)
            next_input = torch.cat([next_img, next_simdepth, next_xy_wrt_cam, next_uv, next_normals], dim=1)
            
            batch_gt_masks, batch_gt_joints_pos, batch_gt_joints_wrt_cam, batch_gt_joints_wrt_rob = batch["next_frame_mask_as_input"].to(device), \
            batch["next_frame_joints_pos"].to(device), batch["next_frame_joints_wrt_cam"].to(device), batch["next_frame_joints_wrt_rob"].to(device)
            middle_time = time.time()
            mask_out, trans_out, quat_out, joint_3d, joint_pos = model(next_input)
            end_time = time.time()
            
#            print("Image Construction Time", middle_time - start_time)
#            print("Model Time", end_time - middle_time)
            
            losses = loss_fn(trans_out, quat_out, joint_pos,joint_3d, mask_out, batch_gt_masks, batch_gt_joints_pos, batch_gt_joints_wrt_cam, batch_gt_joints_wrt_rob)
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print("Total Loss : \n", losses["total_loss"].item())
                print("Mask Loss : \n", losses["mask_loss"].item())
                print("Joint 3d Loss : \n", losses["joint_3d_loss"].item())
                print("Joint Pos Loss : \n", losses["joint_pos_loss"].item())
                print("RT Loss : \n", losses["rt_loss"].item())
                
            if batch_idx % 30 == 0:
                visualize_training_loss(losses, writer, batch_idx, epoch, len(training_loader))
        
        
        # Inference 
        network_inference(model, cfg, epoch, device)
        
        # Visualization
    
    
        # Save Output and Checkpoint
        with torch.no_grad():
            # Save Checkpoint
            save_model(os.path.join(checkpoint_path, "model_{}.pth".format(str(epoch).zfill(3))), epoch, model, optimizer)


if __name__ == "__main__":
    cfg, args = update_config()
    main(cfg)
#    find_seq_data_in_dir("/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/")

















