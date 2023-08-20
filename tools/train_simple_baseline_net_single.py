import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import random
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
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


# spdh
from depth_c2rp.utils.utils import load_camera_intrinsics, set_random_seed, exists_or_mkdir, visualize_inference_results
from depth_c2rp.configs.config import update_config
from depth_c2rp.utils.spdh_utils import reduce_mean, init_spdh_model
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred, log_and_visualize_single
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat
from depth_c2rp.utils.analysis import flat_add_from_pose, add_metrics, batch_add_from_pose, batch_repeat_add_from_pose

# voxel
#from depth_c2rp.datasets.datasets_voxel_ours import Voxel_dataset
from depth_c2rp.datasets.datasets_diffusion_ours import Diff_dataset
#from depth_c2rp.voxel_utils.voxel_network import build_voxel_network, init_voxel_optimizer, load_voxel_model, save_weights, build_voxel_refine_network, load_refine_model, load_optimizer
#from depth_c2rp.voxel_utils.voxel_batch_utils import prepare_data, get_valid_points, get_occ_vox_bound, get_miss_ray, compute_ray_aabb, compute_gt, get_embedding_ours, get_pred, compute_loss, get_embedding, adapt_lr
#from depth_c2rp.voxel_utils.refine_batch_utils import get_pred_refine, compute_refine_loss
from depth_c2rp.models.backbones.dream_hourglass import ResnetSimple, SpatialSoftArgmax

# diffusion
from depth_c2rp.diffusion_utils.simple_baseline_network import build_simple_baseline_network
from depth_c2rp.diffusion_utils.diffusion_losses import get_simple_optimizer, optimization_manager
from depth_c2rp.diffusion_utils.diffusion_ema import ExponentialMovingAverage
from depth_c2rp.diffusion_utils.diffusion_utils import diff_save_weights, diff_load_weights
from depth_c2rp.diffusion_utils.diffusion_sampling import get_sampling_fn
#from inference import network_inference
#from inference_spdh_multi import network_inference

def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main(cfg):
    """
        This code is designed for the whole training. 
    """
    # expect cfg is a dict
    set_random_seed(int(cfg["DIFF_MODEL"]["SEED"]))
    
    assert type(cfg) == dict
    device_ids = [4,5,6,7] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Build DataLoader
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["DIFF_TRAINING"]
    model_cfg = cfg["DIFF_MODEL"]
    training_data_dir = dataset_cfg["TRAINING_ROOT"]
    camera_K = load_camera_intrinsics(os.path.join(training_data_dir, "_camera_settings.json"))
    
    try:
        intrin_aug_params = cfg["voxel_network"]["camera_intrin_aug"]
    except:
        intrin_aug_params = dict()
    
    # Build training and validation set
    training_dataset = Diff_dataset(train_dataset_dir=dataset_cfg["TRAINING_ROOT"],
                                     val_dataset_dir=dataset_cfg["VAL_ROOT"],
                                     real_dataset_dir=dataset_cfg["REAL_ROOT"],
                                     joint_names=[f"panda_joint_3n_{i+1}" for i in range(int(dataset_cfg["NUM_JOINTS"]))],
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
                                     aug_mode=dataset_cfg["AUG"],
                                     change_intrinsic=cfg["CHANGE_INTRINSIC"],
                                     uv_input=False,
                                     intrin_aug_params=intrin_aug_params,
                                     cond_uv_std=train_cfg["COND_UV_STD"],
                                     large_cond_uv_std=train_cfg["LARGE_COND_UV_STD"],
                                     prob_large_cond_uv=train_cfg["PROB_LARGE_COND_UV"],
                                     cond_norm=train_cfg["COND_NORM"],
                                     )
    training_dataset.load_data()    
    training_dataset.train()
    training_loader = DataLoader(training_dataset, batch_size=train_cfg["BATCH_SIZE"], shuffle=True,
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True) 
    
    
    val_dataset = copy.copy(training_dataset)  
    val_dataset.eval()    
    val_loader = DataLoader(val_dataset, batch_size=cfg["DIFF_EVAL"]["BATCH_SIZE"], shuffle=False,
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    real_dataset = copy.copy(training_dataset)
    real_dataset.real()
    real_loader = DataLoader(real_dataset, batch_size=cfg["DIFF_EVAL"]["BATCH_SIZE"], shuffle=False,
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    
    toy_dataset = Diff_dataset(train_dataset_dir=dataset_cfg["VAL_ROOT"],
                                     val_dataset_dir=dataset_cfg["VAL_ROOT"],
                                     real_dataset_dir=dataset_cfg["VAL_ROOT"],
                                     joint_names=[f"panda_joint_3n_{i+1}" for i in range(int(dataset_cfg["NUM_JOINTS"]))],
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
                                     aug_mode=dataset_cfg["AUG"],
                                     change_intrinsic=cfg["CHANGE_INTRINSIC"],
                                     uv_input=False,
                                     intrin_aug_params=intrin_aug_params,
                                     cond_uv_std=train_cfg["COND_UV_STD"],
                                     large_cond_uv_std=train_cfg["LARGE_COND_UV_STD"],
                                     prob_large_cond_uv=train_cfg["PROB_LARGE_COND_UV"],
                                     cond_norm=train_cfg["COND_NORM"],
                                     )
    real_2_dataset = copy.copy(toy_dataset)
    real_2_dataset.real_dataset_dir = Path("/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/2_D415_front_1/")
    real_2_dataset.real()
    real_2_dataset.load_data()
    real_2_dataloader = DataLoader(real_2_dataset, batch_size=cfg["DIFF_EVAL"]["BATCH_SIZE"], shuffle=False,
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    real_3_dataset = copy.copy(toy_dataset)
    real_3_dataset.real_dataset_dir = Path("/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/3_kinect_front_0/")
    real_3_dataset.real()
    real_3_dataset.load_data()    
    real_3_dataloader = DataLoader(real_3_dataset, batch_size=cfg["DIFF_EVAL"]["BATCH_SIZE"], shuffle=False,
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    
    real_4_dataset = copy.copy(toy_dataset)
    real_4_dataset.real_dataset_dir = Path("/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/4_kinect_front_1/")
    real_4_dataset.real()
    real_4_dataset.load_data()   
    real_4_dataloader = DataLoader(real_4_dataset, batch_size=cfg["DIFF_EVAL"]["BATCH_SIZE"], shuffle=False,
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False) 
    
    # Build Recording and Saving Path
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    checkpoint_path = os.path.join(save_path, "CHECKPOINT")
    tb_path = os.path.join(save_path, "TB")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    

    writer = SummaryWriter(tb_path)
    
    # log experiment setting
    param_str = ''
    for key, value in cfg.items():
        if isinstance(value, dict):
            for key1, value1 in value.items():
                param_str += f'{key}.{key1}: {value1}  \n'
        else:
            param_str += f'{key}: {value}  \n'
    writer.add_text('Experiment setting', param_str)


    
    # Build Model 
    model = build_simple_baseline_network(cfg, device)

    start_epoch, global_iter = 0, 0
    print("device", device)

    
    if cfg["RESUME"]:
        print("Resume Model Checkpoint")
        model_ckpt_path = os.path.join(checkpoint_path, "model.pth")    
        print('this_ckpt_path', model_ckpt_path)
        checkpoint = diff_load_weights(model_ckpt_path, device)
        
        state_dict = {}
        for k in checkpoint["model_state_dict"]:
            #print("k", k)
            if k[7:] in model.state_dict():
                state_dict[k[7:]] = checkpoint["model_state_dict"][k]
            if k[17:] in model.state_dict():
                state_dict[k[17:]] = checkpoint["model_state_dict"][k]
            if k in model.state_dict():
                state_dict[k] = checkpoint["model_state_dict"][k]
        ret = model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)                    
    heatmap_model = ResnetSimple(
                                n_keypoints=cfg["DATASET"]["NUM_JOINTS"] * 2,
                                full=cfg["DIFF_TRAINING"]["FULL"],
                                )
    heatmap_model = heatmap_model.to(device)
    
    if cfg["RESUME_HEATMAP"] != "":
        heatmap_model.load_state_dict(torch.load(cfg["RESUME_HEATMAP"], map_location=device)["model"], strict=True)
        print("successfully loading heatmap!")
    
    softargmax_uv = SpatialSoftArgmax(False)
    
    # Build Opimizer
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg["DIFF_MODEL"]["EMA_RATE"])
    optimizer = get_simple_optimizer(cfg, model.parameters())
    optimize_fn = optimization_manager(cfg)
    
    if cfg["RESUME"]:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        ema.load_state_dict(checkpoint['ema'])
        global_iter = checkpoint['step']
    #sampling_fn = get_sampling_fn(cfg, model.sde, model.sampling_shape, model.inverse_scaler, model.sampling_eps, device=device)   
    
    # Build Loss Function
    heatmap_criterion = torch.nn.MSELoss()
    woff_criterion = torch.nn.MSELoss()
    loss_cfg = cfg["LOSS"]
    optim_cfg = cfg["DIFF_OPTIM"]
    num_samples = cfg["DIFF_EVAL"]["NUM_SAMPLES"]
    eval_freq = int(cfg["DIFF_MODEL"]["EVAL_FREQ"])
    simple_criterion = nn.MSELoss(size_average=True).cuda()
    

    print("len_dataloader", len(training_loader))
    visualize_iteration = len(training_loader) // 2
    
    for epoch in tqdm(range(start_epoch + 1, train_cfg["EPOCHS"] + 1)):        
        model.train()
        for batch_idx, batch in enumerate(tqdm(training_loader)):
            global_iter += 1
            if global_iter % optim_cfg["LR_DECAY"] == 0 or global_iter == 1:
                lr_now = lr_decay(optimizer, global_iter, optim_cfg["LR"], optim_cfg["LR_DECAY"], optim_cfg["LR_GAMMA"])
            
            joints_2d = batch["joints_2D_uv"].float().to(device, non_blocking=True)
            joints_3d = batch["joints_3D_Z"].float().to(device, non_blocking=True)
            bs, N, _ = joints_2d.shape
            
            optimizer.zero_grad()
            outputs = model(joints_2d.view(bs, -1))
            outputs = outputs.view(bs, N, -1)
            curr_loss = simple_criterion(outputs, joints_3d)
            curr_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            global_iter += 1
            
            #global_iter = state["step"]
            #optimizer = state["optimizer"]
            #ema = state["ema"]
            
            #print("loss", loss)
            
#            
#            # log cur_loss 
            writer.add_scalar(f'Train/Train Loss', curr_loss.detach().item(), global_iter)
            writer.add_scalar(f"Train/LR", optimizer.param_groups[0]['lr'], global_iter)
                 
        # Save Output and Checkpoint
        with torch.no_grad():
            # Save Checkpoint
            diff_save_weights(epoch, model, optimizer, ema, global_iter, os.path.join(checkpoint_path, f"model.pth"), ) 
            print("save checkpoint !!!")
            if epoch % eval_freq == 0:
                diff_save_weights(epoch, model, optimizer, ema, global_iter, os.path.join(checkpoint_path, "model_{}.pth".format(str(epoch).zfill(5))))

#        
#        #split = {"Validation" : [val_sampler, val_loader],}
        split = {"Validation" : [val_loader], 
                 "1_D415_front_0" : [real_loader], 
                 "real_2_D415_front_1" : [real_2_dataloader],
                 "real_3_kinect_front_0" : [real_3_dataloader],
                 "real_4_kinect_front_1" : [real_4_dataloader],
                 }
        #split = {"Real" : [real_sampler, real_loader],}
        heatmap_model.eval()
        #Validation
        if epoch % eval_freq == 0:
            with torch.no_grad():
                for mode, value in split.items():
                    loader = value[0]
                    model.eval()

                    val_curr_loss = []                  
                    val_add = []
                    
                    for batch_idx, batch in enumerate(tqdm(loader)):
                        curr_loss = 0.0
                        joints_2d = batch["joints_2D_uv"].float().to(device, non_blocking=True)
                        #print("joints_2d.shape", joints_2d.shape)
                        joints_3d = batch["joints_3D_Z"].float().to(device, non_blocking=True)
                        bs, N, _ = joints_2d.shape
#                        joints_2d_yummy = torch.zeros(bs, N, 1).to(device, non_blocking=True)
#                        joints_2d = torch.cat([joints_2d, joints_2d_yummy], dim=-1).float()
#                        
#                        joints_3d_repeat = joints_3d.repeat(num_samples, 1, 1)
#                        joints_2d_repeat = joints_2d.repeat(num_samples, 1, 1)
                        
                        # Generate and save samples
#                        ema.store(model.parameters())
#                        ema.copy_to(model.parameters())
#                        trajs, results = model.sampling_fn(
#                            model.model,
#                            condition=joints_2d_repeat
#                        )  # [b ,j ,3]
#                        ema.restore(model.parameters())
#
#                        # # trajs: [t, b, j, 3], i.e., the pose-trajs
#                        # # results: [b, j, 3], i.e., the end pose of each traj
#                        results = results
#                        
#                        joints_3d_pred_repeat = torch.from_numpy(results).float().to(device)
#                        joints_3d_pred_repeat = joints_3d_pred_repeat.reshape(num_samples, bs, -1, 3) 
#                        joints_3d_pred_repeat = joints_3d_pred_repeat.permute(1, 0, 2, 3) # bs x num_samples x N x 3
#                        joints_3d_pred = torch.mean(joints_3d_pred_repeat, dim=1)
                        outputs = model(joints_2d.view(bs, -1))
                        outputs = outputs.view(bs, N, -1)
                        curr_loss = simple_criterion(outputs, joints_3d)
                        val_curr_loss.append(curr_loss.detach().cpu().numpy())
                        this_add = batch_add_from_pose(outputs.detach().cpu().numpy(), joints_3d.detach().cpu().numpy())
                        val_add = val_add + this_add 
                        #print("results", results.shape)
                        
                        
                        
#                        input_tensor = batch["xyz_img_scale"].float().to(device)
#                        b, _, h, w = input_tensor.shape
#                        heatmap_pred = heatmap_model(input_tensor)[-1]
#                        heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
#                        #print("heatmap_pred.shape", heatmap_pred.shape)
#                        _, c, _, _ = heatmap_pred.shape
#                        
#                        uv_pred = softargmax_uv(heatmap_pred[:, :c//2, :, :])
#                        batch["joints_2D_uv"] = uv_pred
                        
                               
                    val_add_results = add_metrics(val_add, 0.06)
                    print("add results", val_add_results)
                    writer.add_scalar(f"{mode} Metrics/ Loss", np.mean(val_curr_loss), epoch)
                    writer.add_scalar(f"{mode} Metrics/ ADD AUC", val_add_results["add_auc"], epoch)
                    writer.add_scalar(f"{mode} Metrics/ ADD MEDIAN", val_add_results["add_median"], epoch)
                    writer.add_scalar(f"{mode} Metrics/ ADD MEAN", val_add_results["add_mean"], epoch)
                    writer.add_scalar(f"{mode} Metrics/ ADD AUC THRESH", val_add_results["add_auc_thresh"], epoch)
        
        


    
    



if __name__ == "__main__":
    cfg, args = update_config()
    
#    seed = cfg["MODEL"]["SEED"]
#    random.seed(seed)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    if torch.cuda.is_available():
#        torch.backends.cudnn.deterministic = True
#        torch.backends.cudnn.benchmark = False
#    else:
#        torch.backends.cudnn.deterministic = False
#        torch.backends.cudnn.benchmark = True
    
    main(cfg)
#    find_seq_data_in_dir("/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/")

















