import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from torch.utils.data import (DataLoader, Dataset)
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from depth_c2rp.models.heads import FaPNHead 
from depth_c2rp.models.backbones import ResNet
from depth_c2rp.optimizers import get_optimizer, adapt_lr
from depth_c2rp.build import build_model, DataParallel
from depth_c2rp.losses_o6d_reset import get_loss, Calculate_Loss
from depth_c2rp.utils.utils import save_model, load_model, exists_or_mkdir, visualize_training_loss, find_seq_data_in_dir, visualize_training_masks, visualize_inference_results
from depth_c2rp.utils.utils import check_input, visualize_validation_loss, load_camera_intrinsics, visualize_training_lr, set_random_seed, reduce_mean
from depth_c2rp.configs.config import update_config
from depth_c2rp.utils.image_proc import get_nrm
from depth_c2rp.datasets.datasets_o6d import Depth_dataset
#from inference import network_inference
from inference_multi_dr_o6d_reset import network_inference

#def reduce_tensor(losses, num_gpus):
#    losses_copy = {}
#    for key, value in losses.items():
#        rt = value.clone()
#        rt = dist.all_reduce(rt.div_(num_gpus))
#        losses_copy[key] = rt
#    return losses_copy

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def main(cfg):
    """
        This code is designed for the whole training. 
    """
    # expect cfg is a dict
    set_random_seed(int(cfg["MODEL"]["SEED"]))
    
    assert type(cfg) == dict
    device_ids = [4,5,6,7] 
    
    if cfg["LOCAL_RANK"] != -1:
        torch.cuda.set_device(cfg["LOCAL_RANK"])
        device=torch.device("cuda",cfg["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    
    # Build DataLoader
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    training_data_dir = dataset_cfg["TRAINING_ROOT"]
    camera_K = load_camera_intrinsics(os.path.join(training_data_dir, "_camera_settings.json"))
    
    training_data = find_seq_data_in_dir(training_data_dir)
    training_dataset = Depth_dataset(training_data, dataset_cfg["MANIPULATOR"], dataset_cfg["KEYPOINT_NAMES"], dataset_cfg["JOINT_NAMES"], \
    dataset_cfg["INPUT_RESOLUTION"], mask_dict=dataset_cfg["MASK_DICT"], camera_K = camera_K, is_res = dataset_cfg["IS_RES"],device=device,
    twonplus2=dataset_cfg["TWONPLUS2"], num_classes = int(cfg["MODEL"]["MODEL_CLASSES"]))
    #training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=train_cfg["BATCH_SIZE"], shuffle=True, \
    #                                           num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True)
                                               
    train_sampler = DistributedSampler(training_dataset)
    #training_loader = MultiEpochsDataLoader(training_dataset, sampler=train_sampler, batch_size=train_cfg["BATCH_SIZE"],
    #                              num_workers=train_cfg["NUM_WORKERS"], pin_memory=True)
    training_loader = DataLoader(training_dataset, sampler=train_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True)
                                               
    val_data_dir = dataset_cfg["VAL_ROOT"]
    val_data = find_seq_data_in_dir(val_data_dir)
    val_dataset = Depth_dataset(val_data, dataset_cfg["MANIPULATOR"], dataset_cfg["KEYPOINT_NAMES"], dataset_cfg["JOINT_NAMES"], \
    dataset_cfg["INPUT_RESOLUTION"], mask_dict=dataset_cfg["MASK_DICT"], camera_K = camera_K, is_res = dataset_cfg["IS_RES"],device=device,
    twonplus2=dataset_cfg["TWONPLUS2"], num_classes = int(cfg["MODEL"]["MODEL_CLASSES"]))
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_cfg["BATCH_SIZE"], shuffle=True, \
    #                                           num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset)
    #val_loader = MultiEpochsDataLoader(val_dataset, sampler=val_sampler, batch_size=train_cfg["BATCH_SIZE"],
    #                              num_workers=train_cfg["NUM_WORKERS"], pin_memory=True)
    val_loader =DataLoader(val_dataset, sampler=val_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True)
    
    # Build Model
    model_cfg = cfg["MODEL"]
    model = build_model(model_cfg["BACKBONE"], model_cfg["HEAD"], model_cfg["MODEL_CLASSES"], model_cfg["IN_CHANNELS"], \
                        dataset_cfg["NUM_JOINTS"], dataset_cfg["OUTPUT_RESOLUTION"][0], dataset_cfg["OUTPUT_RESOLUTION"][1], model_cfg["ROT_TYPE"])
    if not cfg["RESUME"]:
        model.init_pretrained(model_cfg["PRETRAINED"])
    
    # Build Loss Function
    loss_cfg = cfg["LOSS"]
    loss_fn_names = {"masks" : loss_cfg["NAME"]}
    loss_fn = Calculate_Loss(loss_fn_names,weights=loss_cfg["WEIGHTS"], is_res=dataset_cfg["IS_RES"], device=device, cfg=cfg, rt_loss_type=loss_cfg["RT_LOSS_TYPE"])
    
    # Build Optimizer
    optim_cfg = cfg["OPTIMIZER"]
    #optimizer = get_optimizer(model, optim_cfg["NAME"], optim_cfg["LR"], optim_cfg["WEIGHT_DECAY"])
    
    # Build Recording and Saving Path
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    checkpoint_path = os.path.join(save_path, "CHECKPOINT")
    tb_path = os.path.join(save_path, "TB")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    if dist.get_rank() == 0:
        writer = SummaryWriter(tb_path)
    
    # Load Checkpoint / Resume Training
    start_epoch = 0
    if cfg["RESUME"]:
        print("Resume Model Checkpoint")
        checkpoint_paths = os.listdir(checkpoint_path)
        checkpoint_paths.sort() 
        this_ckpt_path = os.path.join(checkpoint_path, "model.pth")
        print('this_ckpt_path', this_ckpt_path)
        model, start_epoch = load_model(model, this_ckpt_path, optim_cfg["LR"]) 
        print("successfully loaded!")
        
#        model = model.to(device)
#        num_gpus = torch.cuda.device_count()
#        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
#                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
        
      
    model = model.to(device)
    num_gpus = torch.cuda.device_count()
    print("device", device)
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    
    optimizer = get_optimizer(model, optim_cfg["NAME"], optim_cfg["LR"], optim_cfg["WEIGHT_DECAY"]) 

    
    max_iters = len(training_loader) * train_cfg["EPOCHS"]
    print("len_dataloader", len(training_loader))
    
    for epoch in tqdm(range(start_epoch + 1, train_cfg["EPOCHS"] + 1)):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(tqdm(training_loader)):
            pre_process_time = time.time()
            lr_ = adapt_lr(optimizer, epoch, batch_idx, len(training_loader), optim_cfg["LR"], max_iters)
            start_time = time.time()

            
            next_img = batch["next_frame_img_as_input"].to(device)
            next_xy_wrt_cam, next_uv, next_simdepth = batch["next_frame_xy_wrt_cam"].to(device), batch["next_frame_uv"].to(device),\
            batch["next_frame_simdepth_as_input"].to(device)
            next_simdetph_all = batch["next_simdepth_all"].to(device)
            
            
            batch_xyz_rp = batch["next_xyz_rp"].to(device) # B x 1 x 3
            
            batch_gt_masks, batch_gt_joints_pos, batch_gt_joints_wrt_cam = batch["next_frame_mask_as_input"].to(device), \
            batch["next_frame_joints_pos"].to(device), batch["next_frame_joints_wrt_cam"].to(device)
            batch_gt_K, batch_gt_boxes = batch["next_camera_K"].to(device), batch["next_bbox"].to(device)
            batch_gt_trans, batch_gt_quaternion = batch["next_frame_base_trans"].to(device), batch["next_frame_base_quaternion"].to(device)
            batch_gt_2nplus2_wrt_cam = batch["next_joints_2nplus2"].to(device)
            next_normals = get_nrm(next_simdetph_all, camera_K[0][0], batch_gt_boxes[0], dataset_cfg["INPUT_RESOLUTION"])
            
            
            next_input = torch.cat([next_img, next_simdepth, next_xy_wrt_cam, next_uv, next_normals], dim=1).to(device)
            
            middle_time = time.time()
            #print("2:{}".format(torch.cuda.memory_allocated(0)))
            mask_out, poses_out, joint_pos = model(next_input)
            #mask_out, trans_out, quat_out, joint_3d, joint_pos = model.train_step(next_input)
            end_time = time.time()

                       
            losses = loss_fn(poses_out, joint_pos, mask_out, \
                batch_gt_masks, batch_gt_joints_pos, batch_gt_trans, batch_gt_quaternion, batch_gt_joints_wrt_cam, \
                batch_gt_K, batch_gt_boxes, batch_gt_2nplus2_wrt_cam)
            #print("3:{}".format(torch.cuda.memory_allocated(0)))
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
#            if batch_idx == 0:
#                check_input(batch, cfg)
            loss_time = time.time()
            torch.distributed.barrier()
            losses_copy = reduce_mean(losses, num_gpus)
            #print("losses_copy", losses_copy)
            if dist.get_rank() == 0:
                
                visualize_training_loss(losses_copy, writer, batch_idx, epoch, len(training_loader))
                visualize_training_lr(lr_, writer, batch_idx, epoch, len(training_loader))
#                if batch_idx % 50 == 0:
#                    visualize_training_masks(mask_out, writer, device, batch_idx, epoch, len(training_loader))
            visual_time = time.time()
            
#            if dist.get_rank() == 0:
#                if batch_idx > 1:
#                    print("batch_load_time", pre_process_time - pre_time)
                
                
#                print("batch time", middle_time - start_time)
#                print("output time", end_time - middle_time)
#                print("loss time", loss_time - end_time)
#                print("visual_time", visual_time - loss_time)   
            pre_time = visual_time 

        
        
        # Save Output and Checkpoint
        with torch.no_grad():
            # Save Checkpoint
            # save_model(os.path.join(checkpoint_path, "model_{}.pth".format(str(epoch).zfill(3))), epoch, model, optimizer)
            if dist.get_rank() == 0:
                print("save checkpoint")
                save_model(os.path.join(checkpoint_path, "model.pth"), epoch, model, optimizer)
                if epoch % 5 == 0:
                    save_model(os.path.join(checkpoint_path, "model_{}.pth".format(str(epoch).zfill(3))), epoch, model, optimizer)
        
        
        # Validation
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                val_loss, val_mask_loss, val_3d_loss, val_pos_loss, val_rt_loss = [], [], [], [], []
                for val_idx, val_batch in enumerate(tqdm(val_loader)):
                    val_next_img = val_batch["next_frame_img_as_input"].to(device)
                    val_batch_xyz_rp = val_batch["next_xyz_rp"].to(device) # B x 1 x 3
                    val_next_xy_wrt_cam, val_next_uv, val_next_simdepth = val_batch["next_frame_xy_wrt_cam"].to(device), \
                    val_batch["next_frame_uv"].to(device), val_batch["next_frame_simdepth_as_input"].to(device)
                    val_next_simdetph_all = val_batch["next_simdepth_all"].to(device)
                    val_batch_gt_masks, val_batch_gt_joints_pos, val_batch_gt_joints_wrt_cam = val_batch["next_frame_mask_as_input"].to(device), \
                    val_batch["next_frame_joints_pos"].to(device), val_batch["next_frame_joints_wrt_cam"].to(device)
                    
                    val_batch_gt_K, val_batch_gt_boxes = val_batch["next_camera_K"].to(device), val_batch["next_bbox"].to(device)
                    val_batch_gt_trans, val_batch_gt_quaternion = val_batch["next_frame_base_trans"].to(device), val_batch["next_frame_base_quaternion"].to(device)
                    val_batch_gt_2nplus2_wrt_cam = val_batch["next_joints_2nplus2"].to(device)
                    
                    val_next_normals = get_nrm(val_next_simdetph_all, camera_K[0][0], val_batch_gt_boxes[0], dataset_cfg["INPUT_RESOLUTION"])
                    
                    val_next_input = torch.cat([val_next_img, val_next_simdepth, val_next_xy_wrt_cam, val_next_uv, val_next_normals], dim=1)
                    middle_time = time.time() 
                    val_mask_out, val_poses_out, val_joint_pos = model(val_next_input)
                    end_time = time.time()
                    
                    #print("Pre_process Time", start_time - pre_process_time)
                    #print("Image Construction Time", middle_time - start_time)
                    #print("Model Time", end_time - middle_time)
                    #print("val_batch_xyz_rp", val_batch_xyz_rp)
#                    val_losses = loss_fn(val_trans_out, val_quat_out, val_joint_pos,val_mask_out, \
#                        val_batch_gt_masks, val_batch_gt_joints_pos, val_batch_gt_joints_wrt_cam, val_batch_gt_joints_wrt_rob, batch_xyz_rp=val_batch_xyz_rp)
                    val_losses = loss_fn(val_poses_out, val_joint_pos, val_mask_out, \
                    val_batch_gt_masks, val_batch_gt_joints_pos, val_batch_gt_trans, val_batch_gt_quaternion, val_batch_gt_joints_wrt_cam, \
                    val_batch_gt_K, val_batch_gt_boxes,val_batch_gt_2nplus2_wrt_cam)
                    
                    #print(type(val_losses["total_loss"].item()))
                    val_loss.append(val_losses["total_loss"].detach().item())
                    val_mask_loss.append(val_losses["mask_loss"].detach().item())
                    val_3d_loss.append(val_losses["joint_3d_loss"].detach().item())
                    val_pos_loss.append(val_losses["joint_pos_loss"].detach().item())
                    val_rt_loss.append(val_losses["rt_loss"].detach().item())
                if dist.get_rank() == 0:
                    visualize_validation_loss(val_loss, val_mask_loss, val_3d_loss, val_pos_loss, val_rt_loss, writer, epoch)
        
        # Inference
        if epoch % 5 == 0 and dist.get_rank() == 0:  
            ass_add_results, ass_mAP_dict, ori_add_results, ori_mAP_dict, angles_dict = network_inference(model, cfg, epoch, device)
            visualize_inference_results(ass_add_results, ass_mAP_dict, ori_add_results, ori_mAP_dict, angles_dict, writer, epoch)

    
    



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg, args = update_config()
    main(cfg)
#    find_seq_data_in_dir("/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/")

















