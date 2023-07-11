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
from depth_c2rp.diffusion_utils.diffusion_network import build_diffusion_network
from depth_c2rp.diffusion_utils.diffusion_losses import get_optimizer, optimization_manager
from depth_c2rp.diffusion_utils.diffusion_ema import ExponentialMovingAverage
from depth_c2rp.diffusion_utils.diffusion_utils import diff_save_weights, diff_load_weights
from depth_c2rp.diffusion_utils.diffusion_sampling import get_sampling_fn
#from inference import network_inference
#from inference_spdh_multi import network_inference


def main(cfg):
    """
        This code is designed for the whole training. 
    """
    # expect cfg is a dict
    set_random_seed(int(cfg["DIFF_MODEL"]["SEED"]))
    
    assert type(cfg) == dict
    device_ids = [4,5,6,7] 
    
    if cfg["LOCAL_RANK"] != -1:
        torch.cuda.set_device(cfg["LOCAL_RANK"])
        device=torch.device("cuda",cfg["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    
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
                                     change_intrinsic=dataset_cfg["CHANGE_INTRINSIC"],
                                     uv_input=False,
                                     intrin_aug_params=intrin_aug_params,
                                     cond_uv_std=train_cfg["COND_UV_STD"],
                                     large_cond_uv_std=train_cfg["LARGE_COND_UV_STD"],
                                     prob_large_cond_uv=train_cfg["PROB_LARGE_COND_UV"],
                                     )
                                    
    training_dataset.train()
    train_sampler = DistributedSampler(training_dataset)
    training_loader = DataLoader(training_dataset, sampler=train_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True) 
    
    
    val_dataset = copy.copy(training_dataset)  
    val_dataset.eval()    
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=cfg["DIFF_EVAL"]["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    real_dataset = copy.copy(training_dataset)
    real_dataset.real()
    real_sampler = DistributedSampler(real_dataset)
    real_loader = DataLoader(real_dataset, sampler=real_sampler, batch_size=cfg["DIFF_EVAL"]["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    # Build Recording and Saving Path
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    checkpoint_path = os.path.join(save_path, "CHECKPOINT")
    tb_path = os.path.join(save_path, "TB")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    
    if dist.get_rank() == 0:
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

        
        torch.distributed.barrier()
    
    # Build Model 
    model = build_diffusion_network(cfg, device)

    start_epoch, global_iter = 0, 0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    num_gpus = torch.cuda.device_count()
    print("device", device)
    #if num_gpus > 1:
    print('use {} gpus!'.format(num_gpus))
    
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
        ret = model.load_state_dict(state_dict, strict=True)
    
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=True)                                                
    heatmap_model = ResnetSimple(
                                n_keypoints=cfg["DATASET"]["NUM_JOINTS"] * 2,
                                full=cfg["DIFF_TRAINING"]["FULL"],
                                )
    heatmap_model = heatmap_model.to(device)
    heatmap_model = torch.nn.parallel.DistributedDataParallel(heatmap_model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    
    if cfg["RESUME_HEATMAP"] != "":
        heatmap_model.load_state_dict(torch.load(cfg["RESUME_HEATMAP"], map_location=device)["model"], strict=True)
        print("successfully loading heatmap!")
    
    softargmax_uv = SpatialSoftArgmax(False)
    
    # Build Opimizer
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg["DIFF_MODEL"]["EMA_RATE"])
    optimizer = get_optimizer(cfg, model.parameters())
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
    num_samples = cfg["DIFF_EVAL"]["NUM_SAMPLES"]
    eval_freq = int(cfg["DIFF_MODEL"]["EVAL_FREQ"])
    

    print("len_dataloader", len(training_loader))
    visualize_iteration = len(training_loader) // 2
    
    for epoch in tqdm(range(start_epoch + 1, train_cfg["EPOCHS"] + 1)):        
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(tqdm(training_loader)):
            torch.cuda.synchronize()
            
            joints_2d = batch["joints_2D_uv"].float().to(device, non_blocking=True)
            joints_3d = batch["joints_3D_Z"].float().to(device, non_blocking=True)
            bs, N, _ = joints_2d.shape
            joints_2d_yummy = torch.zeros(bs, N, 1).to(device, non_blocking=True)
            joints_2d = torch.cat([joints_2d, joints_2d_yummy], dim=-1).float()
            
            optimizer.zero_grad()
            loss, state = model(joints_3d, joints_2d, optimizer, ema, global_iter)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=global_iter)
            ema.update(model.parameters())
            
            global_iter += 1
            
            #global_iter = state["step"]
            #optimizer = state["optimizer"]
            #ema = state["ema"]
            
            #print("loss", loss)
            
#            
#            # log cur_loss 
            torch.distributed.barrier()
            curr_loss = reduce_mean(loss, num_gpus)

#            
#            
            if dist.get_rank() == 0:
                writer.add_scalar(f'Train/Train Loss', curr_loss.detach().item(), global_iter)
                writer.add_scalar(f"Train/LR", optimizer.param_groups[0]['lr'], global_iter)
                 
        # Save Output and Checkpoint
        with torch.no_grad():
            # Save Checkpoint
            if dist.get_rank() == 0:
            #local_rank = cfg["LOCAL_RANK"]
                diff_save_weights(epoch, model, optimizer, ema, global_iter, os.path.join(checkpoint_path, f"model.pth"), ) 
                print("save checkpoint !!!")
                if epoch % eval_freq == 0:
                    diff_save_weights(epoch, model, optimizer, ema, global_iter, os.path.join(checkpoint_path, "model_{}.pth".format(str(epoch).zfill(5))))

#        
#        #split = {"Validation" : [val_sampler, val_loader],}
        split = {"Real" : [real_sampler, real_loader], "Validation" : [val_sampler, val_loader]}
        #split = {"Real" : [real_sampler, real_loader],}
        heatmap_model.eval()
        #Validation
        if epoch % eval_freq == 0:
            with torch.no_grad():
                for mode, value in split.items():
                    sampler, loader = value[0], value[1]
                    sampler.set_epoch(epoch)
                    model.eval()

                    val_curr_loss = []                  
                    val_add = []
                    
                    for batch_idx, batch in enumerate(tqdm(loader)):
                        curr_loss = 0.0
                        joints_2d = batch["joints_2D_uv"].float().to(device, non_blocking=True)
                        joints_3d = batch["joints_3D_Z"].float().to(device, non_blocking=True)
                        bs, N, _ = joints_2d.shape
                        joints_2d_yummy = torch.zeros(bs, N, 1).to(device, non_blocking=True)
                        joints_2d = torch.cat([joints_2d, joints_2d_yummy], dim=-1).float()
                        
                        joints_3d_repeat = joints_3d.repeat(num_samples, 1, 1)
                        joints_2d_repeat = joints_2d.repeat(num_samples, 1, 1)
                        
                        # Generate and save samples
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        trajs, results = model.module.sampling_fn(
                            model.module.model,
                            condition=joints_2d_repeat
                        )  # [b ,j ,3]
                        ema.restore(model.parameters())

                        # # trajs: [t, b, j, 3], i.e., the pose-trajs
                        # # results: [b, j, 3], i.e., the end pose of each traj
                        results = results
                        
                        joints_3d_pred_repeat = torch.from_numpy(results).float().to(device)
                        joints_3d_pred_repeat = joints_3d_pred_repeat.reshape(num_samples, bs, -1, 3) 
                        joints_3d_pred_repeat = joints_3d_pred_repeat.permute(1, 0, 2, 3) # bs x num_samples x N x 3
                        joints_3d_pred = torch.mean(joints_3d_pred_repeat, dim=1)
                        
                        this_add = batch_add_from_pose(joints_3d_pred.detach().cpu().numpy(), joints_3d.detach().cpu().numpy())
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
                        
                  
                    torch.distributed.barrier()
                    val_add = distributed_concat(torch.from_numpy(np.array(val_add)).to(device), len(sampler.dataset))                 
                    if dist.get_rank() == 0:
                        val_add = val_add.detach().cpu().numpy().tolist()
                        val_add_results = add_metrics(val_add, 0.06)
                        writer.add_scalar(f"{mode} Metrics/ ADD AUC", val_add_results["add_auc"], epoch)
                        writer.add_scalar(f"{mode} Metrics/ ADD MEDIAN", val_add_results["add_median"], epoch)
                        writer.add_scalar(f"{mode} Metrics/ ADD MEAN", val_add_results["add_mean"], epoch)
                        writer.add_scalar(f"{mode} Metrics/ ADD AUC THRESH", val_add_results["add_auc_thresh"], epoch)
        
        


    
    



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
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

















