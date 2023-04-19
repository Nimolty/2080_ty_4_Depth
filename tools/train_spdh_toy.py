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
import copy
from pathlib import Path
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
#from depth_c2rp.models.heads import FaPNHead 
#from depth_c2rp.models.backbones import ResNet
#from depth_c2rp.losses_o6d import get_loss, Calculate_Loss
#from depth_c2rp.utils.utils import save_model, load_model, exists_or_mkdir, visualize_training_loss, find_seq_data_in_dir, visualize_training_masks, visualize_inference_results
#from depth_c2rp.utils.utils import check_input, visualize_validation_loss, load_camera_intrinsics, visualize_training_lr, set_random_seed
#from depth_c2rp.configs.config import update_config
#from depth_c2rp.utils.image_proc import get_nrm

# spdh
from depth_c2rp.utils.utils import load_camera_intrinsics, set_random_seed, exists_or_mkdir
from depth_c2rp.configs.config import update_config
from depth_c2rp.datasets.datasets_toy import Depth_dataset
from depth_c2rp.build import build_toy_spdh_model, build_mode_spdh_model
from depth_c2rp.utils.spdh_utils import load_spdh_model, reduce_mean, save_weights, compute_3n_loss, compute_DX_loss
from depth_c2rp.spdh_optimizers import init_toy_optimizer, adapt_lr
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred, log_and_visualize_single
from depth_c2rp.models.backbones.rest import MLP_TOY
#"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/models/backbones/rest.py"
#from inference import network_inference
from inference_spdh import network_inference


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
    model_cfg = cfg["MODEL"]
    training_data_dir = dataset_cfg["TRAINING_ROOT"]
    camera_K = load_camera_intrinsics(os.path.join(training_data_dir, "_camera_settings.json"))
    
    # Build training and validation set
    training_dataset = Depth_dataset(train_dataset_dir=dataset_cfg["TRAINING_ROOT"],
                                     val_dataset_dir=dataset_cfg["VAL_ROOT"],
                                     joint_names=[f"panda_joint_3n_{i+1}" for i in range(int(dataset_cfg["NUM_JOINTS"]) // 2)],
                                     run=[0],
                                     init_mode="train", 
                                     #noise=self.cfg['noise'],
                                     img_type=dataset_cfg["TYPE"],
                                     raw_img_size=tuple(dataset_cfg["RAW_RESOLUTION"]),
                                     input_img_size=tuple(dataset_cfg["INPUT_RESOLUTION"]),
                                     sigma=dataset_cfg["SIGMA"],
                                     norm_type=dataset_cfg["NORM_TYPE"],
                                     network_input=model_cfg["INPUT_TYPE"],
                                     network_task=model_cfg["TASK"],
                                     depth_range=dataset_cfg["DEPTH_RANGE"],
                                     depth_range_type=dataset_cfg["DEPTH_RANGE_TYPE"],
                                     aug_type=dataset_cfg["AUG_TYPE"],
                                     aug_mode=dataset_cfg["AUG"])
                                    
    training_dataset.train()
    train_sampler = DistributedSampler(training_dataset)
    training_loader = DataLoader(training_dataset, sampler=train_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True)
    
    
    val_dataset = copy.copy(training_dataset)  
    val_dataset.eval()    
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    # Build Recording and Saving Path
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    checkpoint_path = os.path.join(save_path, "CHECKPOINT")
    tb_path = os.path.join(save_path, "TB")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    if dist.get_rank() == 0:
        writer = SummaryWriter(tb_path)
    
    # Build Model 
    if cfg["TOY_NETWORK"] == "Simple_Net":
        kwargs = {"dim" : dataset_cfg["NUM_JOINTS"] // 2 * 3, "h1_dim" :  1024, "out_dim" : 7}
        model = build_mode_spdh_model(kwargs, cfg["TOY_NETWORK"])
    elif cfg["TOY_NETWORK"] == "Transformer_Net":
        kwargs = {"d_inp" : 3, "d_out" : dataset_cfg["NUM_JOINTS"] // 6, "d_model" : 128, "n_k":dataset_cfg["NUM_JOINTS"] // 2
                  }
        model = build_mode_spdh_model(kwargs, cfg["TOY_NETWORK"])
    
    
    start_epoch, global_iter = 0, 0
    model = model.to(device)
    num_gpus = torch.cuda.device_count()
    print("device", device)
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    
    # Build Opimizer
    optimizer, scheduler = init_toy_optimizer(model, cfg)
    base_lr = cfg["OPTIMIZER"]["LR"]
    
    # Load Trained Model
    
    if cfg["RESUME"]:
        print("Resume Model Checkpoint")
        checkpoint_paths = os.listdir(checkpoint_path)
        checkpoint_paths.sort() 
        this_ckpt_path = os.path.join(checkpoint_path, "model.pth")
        print('this_ckpt_path', this_ckpt_path)
        model, optimizer, scheduler, start_epoch, global_iter = load_spdh_model(model, optimizer, scheduler, this_ckpt_path, device)
        print("successfully loaded!")
    

    # Build Loss Function
    toy_criterion = torch.nn.L1Loss()
    loss_cfg = cfg["LOSS"]
#    loss_fn_names = {"masks" : loss_cfg["NAME"]}
#    loss_fn = Calculate_Loss(loss_fn_names,weights=loss_cfg["WEIGHTS"], is_res=dataset_cfg["IS_RES"], device=device, cfg=cfg, rt_loss_type=loss_cfg["RT_LOSS_TYPE"])
    

    print("len_dataloader", len(training_loader))
    max_iters = len(training_loader) * train_cfg["EPOCHS"]
    visualize_iteration = len(training_loader) // 2
    print("THREE_D_NORM", cfg["THREE_D_NORM"])
    
    for epoch in tqdm(range(start_epoch + 1, train_cfg["EPOCHS"] + 1)):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(tqdm(training_loader)):
            joints_3d = batch['joints_3D_Z'].to(device).float()
            joints_1d = batch["joints_7"].to(device).float()

            if cfg["TOY_NETWORK"] == "Simple_Net":
                if cfg["THREE_D_NORM"]:
                    joints_3d = joints_3d - joints_3d[:, :1, :]
                if cfg["THREE_D_NOISE"]:
                    joints_3d_randn = torch.randn_like(joints_3d) # B x N x 3
                    joints_3d_randn /= torch.norm(joints_3d_randn, dim=-1, keepdim=True)
                    joints_3d += joints_3d_randn * float(cfg["THREE_D_NOISE"])
                    
                outputs = model(torch.flatten(joints_3d, 1))
            elif cfg["TOY_NETWORK"] == "Transformer_Net":
                outputs = model(joints_3d, joints_3d, joints_3d)
            else:
                raise ValueError
            
            if loss_cfg["THREE_D_LOSS_TYPE"] == "edm":
                joints_3d_pred = compute_3n_loss(outputs[:, :, None], joints_1d.device)
                #print("joints_3d_pred", joints_3d_pred.shape)
                #print("joints_3d", joints_3d.shape)
                edm_loss = compute_DX_loss(joints_3d_pred, joints_3d)
                curr_loss = loss_cfg["Q_WEIGHTS"] * toy_criterion(outputs, joints_1d) + loss_cfg["EDM_WEIGHTS"] * edm_loss
            else:
                curr_loss = toy_criterion(outputs, joints_1d)

            
            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step() 
            
            # log cur_loss 
            torch.distributed.barrier()
            curr_loss = reduce_mean(curr_loss, num_gpus)
            
            if dist.get_rank() == 0:
#                print("joints_1d_loss", curr_loss - edm_loss)
#                print("edm loss", edm_loss)
                if loss_cfg["THREE_D_LOSS_TYPE"] == "edm": 
                    writer.add_scalar(f'Train/Train All Loss', curr_loss.detach().item(), global_iter)
                    writer.add_scalar(f'Train/Train EDM Loss', (loss_cfg["EDM_WEIGHTS"] * edm_loss).detach().item(), global_iter)
                    writer.add_scalar(f'Train/Train Loss', (curr_loss - loss_cfg["EDM_WEIGHTS"] * edm_loss).detach().item(), global_iter)
                else:
                    writer.add_scalar(f'Train/Train Loss', curr_loss.detach().item(), global_iter)
                writer.add_scalar(f"Train/LR", optimizer.param_groups[0]['lr'], global_iter)
                   
#            if batch_idx % visualize_iteration == 0 and dist.get_rank() == 0:
#                heatmap_pred, joints_3d_pred = get_joint_3d_pred(heatmap_pred, cfg, h, w, c, input_K)
#                joints_3d_pred = joints_3d_pred[:8]
#                joints_3d_gt = joints_3d.clone().cpu().numpy()[:8]
#                gt_images = batch['depthvis'].clone().numpy()[:8]
#                K = batch['K_depth'].clone().numpy()[:8]
#                gt_images, pred_images, true_blend_uv, true_blend_uz, pred_blend_uv, pred_blend_uz = get_blended_images(gt_images, K, joints_3d_gt, joints_3d_pred, device, heatmap_pred, heatmap_gt)
#                gt_results, pred_results = [gt_images], [pred_images]
#                true_blends_UV, pred_blends_UV = [true_blend_uv], [pred_blend_uv]
#                true_blends_UZ, pred_blends_UZ = [true_blend_uz], [pred_blend_uz]
#                log_and_visualize_single(writer, global_iter, gt_results, pred_results,true_blends_UV, pred_blends_UV,true_blends_UZ, pred_blends_UZ)
            
            
            global_iter += 1
            #adapt_lr(optimizer, global_iter, base_lr, max_iters)
                

        # Save Output and Checkpoint
        with torch.no_grad():
            # Save Checkpoint
            if dist.get_rank() == 0:
                print("save checkpoint")
                save_weights(os.path.join(checkpoint_path, "model.pth"), epoch, global_iter, model, optimizer, scheduler, cfg)
                if epoch % 5 == 0:
                    save_weights(os.path.join(checkpoint_path, "model_{}.pth".format(str(epoch).zfill(3))), epoch, global_iter, model, optimizer, scheduler, cfg)
        
        
#        # Validation
#        if epoch % 1 == 0:
#            with torch.no_grad():
#                val_sampler.set_epoch(epoch)
#                model.eval()
#                val_curr_loss = []
#                for batch_idx, batch in enumerate(tqdm(val_loader)):
#                    joints_3d = batch['joints_3D_Z'].to(device).float()
#                    heatmap_gt = batch['heatmap_25d'].to(device).float()
#                    input_K = batch['K_depth'].to(device).float()
#                    input_fx = batch['K_depth'].to(device).float()[:, 0, 0]
#                    input_fy = batch['K_depth'].to(device).float()[:, 1, 1]
#                    
#                    if model_cfg["INPUT_TYPE"] == "XYZ":
#                        input_tensor = batch['xyz_img'].to(device).float()
#                    else:
#                        raise ValueError
#                    
#                    outputs = model(input_tensor)
#                    b, c, h, w = heatmap_gt.size()
#                    if model_cfg["NAME"] == "stacked_hourglass":
#                        heatmap_pred = outputs['heatmaps']
#                        curr_loss = heatmap_criterion((F.interpolate(heatmap_pred[0], (h, w), mode='bicubic',
#                                                                  align_corners=False) + 1) / 2., heatmap_gt)
#                        for j in range(1, len(heatmap_pred)):
#                            curr_loss += heatmap_criterion((F.interpolate(heatmap_pred[j], (h, w), mode='bicubic',
#                                                                       align_corners=False) + 1) / 2., heatmap_gt)
#                    elif "dreamhourglass" in model_cfg["NAME"]:
#                        heatmap_pred = outputs[-1]
#                        curr_loss = heatmap_criterion((F.interpolate(heatmap_pred, (h, w), mode='bicubic',
#                                                                  align_corners=False) + 1) / 2., heatmap_gt)
#                    else:
#                        raise ValueError
#                    val_curr_loss.append(curr_loss.detach().item())
#                if dist.get_rank() == 0:
#                    writer.add_scalar(f'Validation/Validation Loss', np.mean(val_curr_loss), epoch)
#                
#        
#        # Inference
#        if epoch % 1 == 0 and dist.get_rank() == 0:  
#            ass_add_results, ass_mAP_dict = network_inference(model, cfg, epoch, device)
#            visualize_inference_results(ass_add_results, ass_mAP_dict, writer, epoch)
        
        scheduler.step(epoch + 1)

    
    



if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')
    cfg, args = update_config()
    main(cfg)
#    find_seq_data_in_dir("/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/")

















