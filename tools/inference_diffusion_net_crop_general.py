import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import cv2
import json
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
from depth_c2rp.utils.spdh_utils import reduce_mean, init_spdh_model, compute_kps_joints_loss
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred, log_and_visualize_single
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat
from depth_c2rp.utils.analysis import flat_add_from_pose, add_metrics, add_from_pose, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles, batch_outlier_removal_pose, batch_repeat_add_from_pose, batch_repeat_mAP_from_pose, batch_repeat_acc_from_joint_angles, batch_pck_from_pose, pck_metrics
from depth_c2rp.utils.rgb_utils import batch_affine_transform_pts

from depth_c2rp.models.backbones.dream_hourglass import ResnetSimple, SpatialSoftArgmax
#from depth_c2rp.datasets.datasets_voxel_ours import Voxel_dataset
from depth_c2rp.datasets.datasets_diffusion_inference_crop import Diff_dataset
#from depth_c2rp.datasets.datasets_diffusion_inference_ours import Diff_dataset

#from depth_c2rp.voxel_utils.voxel_network import build_voxel_simple_network, load_simplenet_model

# diffusion
from depth_c2rp.diffusion_utils.diffusion_network_general import build_diffusion_network, build_simple_network, load_simplenet_model
from depth_c2rp.diffusion_utils.diffusion_losses_general import get_optimizer, optimization_manager
from depth_c2rp.diffusion_utils.diffusion_ema import ExponentialMovingAverage
from depth_c2rp.diffusion_utils.diffusion_utils_general import diff_save_weights, diff_load_weights
from depth_c2rp.diffusion_utils.diffusion_sampling_general import get_sampling_fn

def main(cfg, mAP_thresh=[0.02, 0.11, 0.01], add_thresh=0.06,angles_thresh=[2.5, 30.0, 2.5]):
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
    eval_cfg = cfg["DIFF_EVAL"]
    training_data_dir = dataset_cfg["TRAINING_ROOT"]
    val_dataset_dir=dataset_cfg["VAL_ROOT"]
    print("val_dataset", dataset_cfg["VAL_ROOT"])
    real_dataset_dir=dataset_cfg["REAL_ROOT"],
    camera_K = load_camera_intrinsics(os.path.join(training_data_dir, "_camera_settings.json"))
    
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
                                     output_img_size=tuple(dataset_cfg["OUTPUT_RESOLUTION"]),
                                     sigma=dataset_cfg["SIGMA"],
                                     norm_type=dataset_cfg["NORM_TYPE"],
                                     network_input=model_cfg["INPUT_TYPE"],
                                     network_task=model_cfg["TASK"],
                                     depth_range=dataset_cfg["DEPTH_RANGE"],
                                     depth_range_type=dataset_cfg["DEPTH_RANGE_TYPE"],
                                     aug_type=dataset_cfg["AUG_TYPE"],
                                     aug_mode=False,
                                     change_intrinsic=cfg["CHANGE_INTRINSIC"],
                                     uv_input=False,
                                     cond_uv_std=train_cfg["COND_UV_STD"],
                                     cond_norm=train_cfg["COND_NORM"],
                                     mean=model_cfg["MEAN"],
                                     std=model_cfg["STD"],
                                     )
                                    
    training_dataset.train()
    train_sampler = DistributedSampler(training_dataset)
    training_loader = DataLoader(training_dataset, sampler=train_sampler, batch_size=eval_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True) 
    
    
    val_dataset = copy.copy(training_dataset)  
    val_dataset.eval()    
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=eval_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    real_dataset = copy.copy(training_dataset)
    real_dataset.real()
    real_sampler = DistributedSampler(real_dataset)
    real_loader = DataLoader(real_dataset, sampler=real_sampler, batch_size=eval_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    # Build Recording and Saving Path
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    checkpoint_path = os.path.join(save_path, "CHECKPOINT")
    tb_path = os.path.join(save_path, "TB")
    heatmap_id = cfg["RESUME_HEATMAP"].split('/')[-2][8:11]
    pred_mask_flag = cfg["PRED_MASK"]
    results_path = os.path.join(save_path, f"NUMERICAL_RESULTS_{heatmap_id}_PRED_MASK_{pred_mask_flag}")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    exists_or_mkdir(results_path)

    
    # Inference 
    ass_add = []
    ass_per_add = []
    ass_mAP = []
    angles_acc = []
    kps_add = []
    kps_mAP = []
    uv_pck = []
    z_pck = []
    acc = []
    
    
    # lst
    joints_angle_pred_lst = []
    joints_3d_pred_lst = []
    joints_3d_pred_repeat_lst = []
    pose_pred_lst = []
    kps_pred_lst = []
    joints_angle_gt_lst = []
    joints_3d_gt_lst = []
    pose_gt_lst = []
    kps_gt_lst = []
    joints_3d_err_lst = []
    pred_z = []
    gt_z = []
    
    # Evaludation Thresholds
    start_thresh_mAP, end_thresh_mAP, interval_mAP = mAP_thresh
    thresholds = np.arange(start_thresh_mAP, end_thresh_mAP, interval_mAP)
    thresh_length = len(thresholds)
    start_angle_acc, end_angle_acc, interval_acc = angles_thresh
    acc_thresholds = np.arange(start_angle_acc, end_angle_acc, interval_acc)
    
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
    simplenet_model = build_simple_network(cfg)
    
    if cfg["RESUME"]:
        print("Resume Model Checkpoint")
        resume_checkpoint = cfg["RESUME_CHECKPOINT"]
        if resume_checkpoint:
            model_ckpt_path = os.path.join(checkpoint_path, resume_checkpoint)
        else:
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

    start_epoch,global_iter = 0, 0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    simplenet_model = simplenet_model.to(device)
    
    num_gpus = torch.cuda.device_count()
    print("device", device)
    #if num_gpus > 1:
    print('use {} gpus!'.format(num_gpus))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    simplenet_model = torch.nn.parallel.DistributedDataParallel(simplenet_model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
                                                                                          
#    heatmap_model = ResnetSimple(
#                                n_keypoints=cfg["DATASET"]["NUM_JOINTS"] * 2,
#                                full=cfg["DIFF_TRAINING"]["FULL"],
#                                )

    heatmap_model = ResnetSimple(
                                n_keypoints=cfg["DATASET"]["NUM_JOINTS"],
                                full=cfg["DIFF_TRAINING"]["FULL"],
                                )

    heatmap_model = heatmap_model.to(device)
    heatmap_model = torch.nn.parallel.DistributedDataParallel(heatmap_model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    
    if cfg["RESUME_SIMPLENET"] != "":
        simplenet_model = load_simplenet_model(simplenet_model, cfg["RESUME_SIMPLENET"], device)
    if cfg["RESUME_HEATMAP"] != "":
        heatmap_model.load_state_dict(torch.load(cfg["RESUME_HEATMAP"], map_location=device)["model"], strict=True)
        print("successfully loading heatmap!")
    
    softargmax_uv = SpatialSoftArgmax(False)

    # Build Opimizer
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg["DIFF_MODEL"]["EMA_RATE"])
    optimizer = get_optimizer(cfg, model.parameters())
    
    
    if cfg["RESUME"]:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        ema.load_state_dict(checkpoint['ema'])
        global_iter = checkpoint['step']
    

    # Build Loss Function
    heatmap_criterion = torch.nn.MSELoss()
    woff_criterion = torch.nn.MSELoss()
    loss_cfg = cfg["LOSS"]
    num_samples = cfg["DIFF_EVAL"]["NUM_SAMPLES"]
    method = cfg["DIFF_SAMPLING"]["METHOD"]

    print("len_dataloader", len(training_loader))
    print("num_samples", num_samples)
#    with torch.no_grad():
    gt_angle_flag = train_cfg["GT_ANGLE_FLAG"]
    pred_2d_flag = cfg["PRED_2D_FLAG"]
    cond_norm_flag = train_cfg["COND_NORM"]
    #split = {"Real" : [real_sampler, real_loader, real_dataset_dir], "Validation" : [val_sampler, val_loader, val_dataset_dir]}
    #split = {"Validation" : [val_sampler, val_loader, val_dataset_dir]}
    split = {"Real" : [real_sampler, real_loader, real_dataset_dir],}
    heatmap_model.eval()
    model.eval()
    simplenet_model.eval()
    time_list = []
    time_list2 = []
    meta_json = {}
    results_list = []
    uv_pred_list = []
    uv_gt_list = []
    uv_pck_list = []
    uv_kps_pck_list = []
    depth_path_lst = []
    info_path = os.path.join(save_path, "INFERENCE_LOGS", str(start_epoch).zfill(5))
    if dist.get_rank() == 0:
        exists_or_mkdir(info_path)
    
    infer_prob = cfg.get("INFER_PROB", 0.0)
    print("infer_prob", infer_prob)
    with torch.no_grad():
        for mode, value in split.items():
            sampler, loader = value[0], value[1]
            sampler.set_epoch(start_epoch)
            #sampler.set_epoch(0)
            
            for batch_idx, batch in enumerate(tqdm(loader)):
                batch_json = {}
                curr_loss = 0.0
                
#                if batch_idx > 1:
#                    break
                
                #input_tensor = batch["xyz_img_scale"].float().to(device)
                #xyz_img = batch["xyz_img"].detach().cpu().numpy()
                joints_3d_gt = batch['joints_3D_Z'].to(device, non_blocking=True).float()
                joints_kps_3d_gt = batch["joints_3D_kps"].to(device, non_blocking=True).float()
                joints_1d_gt = batch["joints_7"].to(device).float()
                pose_gt = batch["R2C_Pose"].to(device).float()
                intrinsic = batch["intrinsic"].to(device).float()
                fx, fy = intrinsic[0, 0, 0], intrinsic[0, 1, 1]
                cx, cy = intrinsic[0, 0, 2], intrinsic[0, 1, 2]
                bbox = batch["bbox"].detach().cpu().numpy() # B x 4
                trans_output_inverse = batch["trans_output_inverse"].detach().cpu().numpy()
                input_uv_off = batch["joints_2D_output_off"].to(device).float()
                joints_2D_output_ind = batch["joints_2D_output_ind"].to(device).type(torch.int64)
                depth_path = batch["meta_path"]
                
                depth_path_lst += depth_path
                
                #depth_img_vis = batch["depthvis"].detach().cpu().numpy()
                #depth_path = batch["depth_path"]
                #batch_json["depth_path"] = depth_path[0]

                
                
                torch.cuda.synchronize()
                t1 = time.time()
                if pred_2d_flag:
                    input_tensor = batch["input_tensor"].float().to(device)
                    uv_gt = batch['joints_2D'].to(device).float()
                    _, _, h, w = input_tensor.shape
                    heatmap_pred = heatmap_model(input_tensor)[-1]
                    if heatmap_pred.shape[2] != dataset_cfg["OUTPUT_RESOLUTION"][1] and heatmap_pred.shape[3] != dataset_cfg["OUTPUT_RESOLUTION"][0]: 
                        heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
                    
                    #uv_pred = softargmax_uv(heatmap_pred[:, :c//2, :, :])   
                    print(heatmap_pred.shape)
                    uv_pred = softargmax_uv(heatmap_pred)
                    n_kp = uv_pred.shape[1]
                    uv_pred = batch_affine_transform_pts(uv_pred.detach().cpu().numpy(), trans_output_inverse)  
                    uv_pred[:, :, 0] += np.repeat(bbox[:, 1:2], n_kp, axis=1)
                    uv_pred[:, :, 1] += np.repeat(bbox[:, 0:1], n_kp, axis=1)
                    uv_pred = torch.from_numpy(uv_pred).float().to(device)
                    
                    uv_pred_list.append(uv_pred)
                    uv_gt_list.append(uv_gt)
                    uv_pck = batch_pck_from_pose(uv_gt.detach().cpu().numpy(), uv_pred.detach().cpu().numpy())
                    uv_kps_pck = torch.linalg.norm((uv_gt-uv_pred), dim=-1) # B x N
                    uv_kps_pck_list.append(uv_kps_pck.detach().cpu().numpy().tolist())
                    uv_pck_list.append(uv_pck)
                    
                    #print("uv_dis", uv_pred - uv_gt)
                    
                    
                    if cond_norm_flag:
                        batch["joints_2D_cond_norm"][:, :, 0] = (uv_pred[:, :, 0] - cx) / fx
                        batch["joints_2D_cond_norm"][:, :, 1] = (uv_pred[:, :, 1] - cy) / fy
                    else:
                        raise NotImplementedError
                
                
                joints_2d = batch["joints_2D_cond_norm"].float().to(device, non_blocking=True)
#                joints_3d = batch["joints_3D_Z"].float().to(device, non_blocking=True)
                bs, N, _ = joints_2d.shape
                joints_2d_yummy = torch.zeros(bs, N, 1).to(device, non_blocking=True)
                joints_2d = torch.cat([joints_2d, joints_2d_yummy], dim=-1).float()
                
                joints_3d_gt_repeat = joints_3d_gt.repeat(num_samples, 1, 1)
                joints_2d_repeat = joints_2d.repeat(num_samples, 1, 1)
                joints_1d_gt_repeat = joints_1d_gt.repeat(num_samples, 1)
                pose_gt_repeat = pose_gt.repeat(num_samples, 1, 1)
                joints_kps_3d_gt_repeat = joints_kps_3d_gt.repeat(num_samples, 1, 1)
                        
                # Generate and save samples
                #ema.store(model.parameters())
                infer_prob_input = torch.ones(bs * num_samples)[..., None].float().to(device, non_blocking=True) * infer_prob
                #infer_prob = torch.bernoulli(infer_prob)
                
                ema.copy_to(model.parameters())
                
                joints_embed, infer_prob_input = model.module.model.fix_joints(joints=joints_1d_gt_repeat, prob=infer_prob_input)
                cond_embed = model.module.model.fix_cond(joints_2d_repeat)
                #print("infer_prob_input", infer_prob_input)
                
                trajs, results = model.module.sampling_fn(
                            model.module.model(joints_embed, cond_embed, infer_prob_input),
                            shape=joints_3d_gt_repeat.shape
                        )  # [b ,j ,3]
                
                
                #print("t2 - t1", t2 - t1)
                #ema.restore(model.parameters())
                
                
                joints_3d_pred_repeat = results[-1].to(device)
                
                joints_3d_pred_repeat = joints_3d_pred_repeat.reshape(num_samples, bs, -1, 3) 
                joints_3d_pred_repeat = joints_3d_pred_repeat.permute(1, 0, 2, 3) # bs x num_samples x N x 3
                #print("joints_3d_pred_repeat", joints_3d_pred_repeat[0])
                joints_3d_pred = torch.mean(joints_3d_pred_repeat, dim=1)
                
                joints_3d_pred_repeat = joints_3d_pred_repeat.permute(0, 2, 1, 3) # bs x N x num_samples x 3

                #joints_angle_pred_repeat, pose_pred_repeat = simplenet_model(joints_3d_pred_repeat.clone(), joints_1d_gt_repeat[..., None], gt_angle_flag) 
                joints_angle_pred, pose_pred = simplenet_model(joints_3d_pred.clone(), joints_1d_gt[..., None], gt_angle_flag) 
                torch.cuda.synchronize()
                t2 = time.time()
                print("t2 - t1", t2 - t1)
                
                results_repeat = torch.cat(trajs,dim=0).reshape(-1, num_samples, bs, N, 3) # steps x num_samples x bs x N x 3
                results_repeat = results_repeat.permute(2, 1, 0, 3, 4) # bs x num_samples x steps x N x 3

                
                if gt_angle_flag:
                    joints_angle_pred = joints_1d_gt[..., None]  
                
                joints_angle_pred_lst.append(joints_angle_pred)
                joints_angle_gt_lst.append(joints_1d_gt)
                joints_3d_pred_lst.append(joints_3d_pred)
                joints_3d_gt_lst.append(joints_3d_gt)
                pose_pred_lst.append(pose_pred)
                pose_gt_lst.append(pose_gt) 
                joints_3d_pred_repeat_lst.append(joints_3d_pred_repeat)
                results_list.append(results_repeat)
                
                
                #joints_kps_3d_pred_repeat = compute_kps_joints_loss(pose_pred_repeat[:, :3, :3], (pose_pred_repeat[:, :3, 3])[:, :, None], joints_angle_pred_repeat, device)  
                joints_kps_3d_pred = compute_kps_joints_loss(pose_pred[:, :3, :3], (pose_pred[:, :3, 3])[:, :, None], joints_angle_pred, device)  
                
                
                #kps_add = kps_add + batch_repeat_add_from_pose(joints_kps_3d_pred_repeat.detach().cpu().numpy(), joints_kps_3d_gt_repeat.detach().cpu().numpy(), bs, num_samples)
                #kps_mAP.append(batch_repeat_mAP_from_pose(joints_kps_3d_pred_repeat.detach().cpu().numpy(), joints_kps_3d_gt_repeat.detach().cpu().numpy(),bs, num_samples, thresholds))
                kps_add = kps_add + batch_add_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy())
                kps_mAP.append(batch_mAP_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy(), thresholds))
        
                
                kps_pred_lst.append(joints_kps_3d_pred)
                kps_gt_lst.append(joints_kps_3d_gt)
                                
                #ass_add_mean = batch_repeat_add_from_pose(joints_3d_pred_repeat.detach().cpu().numpy(), joints_3d_gt_repeat.detach().cpu().numpy(), bs, num_samples)
                #ass_add = ass_add + ass_add_mean
                
                #ass_mAP_mean = batch_repeat_mAP_from_pose(joints_3d_pred_repeat.detach().cpu().numpy(), joints_3d_gt_repeat.detach().cpu().numpy(), bs,num_samples, thresholds) # 
                #ass_mAP.append(ass_mAP_mean)
                joints_3d_pred_np, joints_3d_gt_np = joints_3d_pred.detach().cpu().numpy(), joints_3d_gt.detach().cpu().numpy()
                ass_add_mean = batch_add_from_pose(joints_3d_pred_np, joints_3d_gt_np)
                ass_add = ass_add + ass_add_mean
                
                ass_per_add_mean = np.linalg.norm((joints_3d_pred_np - joints_3d_gt_np), axis=-1) # B x N
                ass_per_add.append(ass_per_add_mean.tolist())
                
                ass_mAP_mean = batch_mAP_from_pose(joints_3d_pred.detach().cpu().numpy(), joints_3d_gt.detach().cpu().numpy(), thresholds) # 
                ass_mAP.append(ass_mAP_mean)
                
                #angles_acc_mean = batch_repeat_acc_from_joint_angles(joints_angle_pred_repeat.detach().cpu().numpy(), joints_1d_gt_repeat.detach().cpu().numpy()[:, :, None], bs, num_samples, acc_thresholds)
                #print("angles_acc_mean.shape", np.array(angles_acc_mean).shape)
                #angles_acc.append(angles_acc_mean)
                angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred.detach().cpu().numpy(), joints_1d_gt.detach().cpu().numpy()[:, :, None], acc_thresholds)
                #print("angles_acc_mean.shape", np.array(angles_acc_mean).shape)
                angles_acc.append(angles_acc_mean)
                
            torch.distributed.barrier()
            angles_acc = np.concatenate(angles_acc, axis=0)
            ass_mAP = np.concatenate(ass_mAP, axis=0)
            kps_mAP = np.concatenate(kps_mAP, axis=0)
            ass_per_add = np.concatenate(ass_per_add, axis=0)
            
            
#            print(uv_pred_list)
            if pred_2d_flag:
                uv_pck_list = np.concatenate(uv_pck_list, axis=0) 
                uv_pck_list = distributed_concat(torch.from_numpy(uv_pck_list).to(device), len(sampler.dataset))
                uv_kps_pck_list = np.concatenate(uv_kps_pck_list, axis=0) 
                uv_kps_pck_list = distributed_concat(torch.from_numpy(uv_kps_pck_list).to(device), len(sampler.dataset))
                uv_pred_list = distributed_concat(torch.cat(uv_pred_list,dim=0), len(sampler.dataset))
                uv_gt_list = distributed_concat(torch.cat(uv_gt_list,dim=0), len(sampler.dataset))
                
                #print("uv_pred_list.shape", uv_pred_list.shape)
            ass_add = distributed_concat(torch.from_numpy(np.array(ass_add)).to(device), len(sampler.dataset))
            ass_per_add = distributed_concat(torch.from_numpy(np.array(ass_per_add)).to(device), len(sampler.dataset))
            ass_mAP = distributed_concat(torch.from_numpy(np.array(ass_mAP)).to(device), len(sampler.dataset))
            angles_acc = distributed_concat(torch.from_numpy(np.array(angles_acc)).to(device), len(sampler.dataset))
            kps_add  = distributed_concat(torch.from_numpy(np.array(kps_add)).to(device), len(sampler.dataset))
            kps_mAP = distributed_concat(torch.from_numpy(np.array(kps_mAP)).to(device), len(sampler.dataset))
            
            joints_3d_pred_gather = distributed_concat(torch.cat(joints_3d_pred_lst, dim=0), len(sampler.dataset))
            joints_angle_pred_gather = distributed_concat(torch.cat(joints_angle_pred_lst, dim=0), len(sampler.dataset))
            pose_pred_gather = distributed_concat(torch.cat(pose_pred_lst, dim=0), len(sampler.dataset))
            joints_3d_gt_gather = distributed_concat(torch.cat(joints_3d_gt_lst, dim=0), len(sampler.dataset))
            joints_angle_gt_gather = distributed_concat(torch.cat(joints_angle_gt_lst, dim=0), len(sampler.dataset))
            pose_gt_gather = distributed_concat(torch.cat(pose_gt_lst, dim=0), len(sampler.dataset))
            
            kps_pred_gather = distributed_concat(torch.cat(kps_pred_lst, dim=0), len(sampler.dataset))
            kps_gt_gather = distributed_concat(torch.cat(kps_gt_lst, dim=0), len(sampler.dataset))
            
            joints_3d_pred_repeat_gather = distributed_concat(torch.cat(joints_3d_pred_repeat_lst, dim=0), len(sampler.dataset))
            results_list = distributed_concat(torch.cat(results_list, dim=0), len(sampler.dataset))

            
            ass_add = ass_add.detach().cpu().numpy().tolist()
            ass_per_add = ass_per_add.detach().cpu().numpy().tolist()
            ass_mAP = ass_mAP.detach().cpu().numpy().tolist()
            angles_acc = angles_acc.detach().cpu().numpy().tolist()
            kps_add = kps_add.detach().cpu().numpy().tolist()
            kps_mAP = kps_mAP.detach().cpu().numpy().tolist()
            pose_gt_gather = pose_gt_gather.detach().cpu().numpy().tolist()
            if pred_2d_flag:
                uv_pred_list = uv_pred_list.detach().cpu().numpy()
                uv_gt_list = uv_gt_list.detach().cpu().numpy()
                uv_pck_list = uv_pck_list.detach().cpu().numpy()
                uv_kps_pck_list = uv_kps_pck_list.detach().cpu().numpy()
                pck_results = pck_metrics(uv_pck_list)
            
            
            angles_results = np.round(np.mean(angles_acc,axis=0)*100, 2)
            angles_dict = dict()
            kps_add_results = add_metrics(kps_add, add_thresh)
            kps_mAP_results = np.round(np.mean(kps_mAP, axis=0) * 100, 2)
            kps_mAP_dict = dict()
            ass_add_results = add_metrics(ass_add, add_thresh)
            ass_mAP_results = np.round(np.mean(ass_mAP, axis=0) * 100, 2)
            ass_mAP_dict = dict()
            
            change_intrinsic_flag = cfg["CHANGE_INTRINSIC"]
            real_name = dataset_cfg["REAL_ROOT"].split('/')[-2]
            #real_name = dataset_cfg["VAL_ROOT"].split('/')[-2]
            file_name = os.path.join(results_path, f"Epoch_{str(start_epoch).zfill(5)}_{real_name}_infer_prob_{str(infer_prob)}_angle_{gt_angle_flag}_ns_{num_samples}_pred2d_{pred_2d_flag}_{method}.txt")
            path_meta = os.path.join(info_path, f"Epoch_{str(start_epoch).zfill(5)}_{real_name}_infer_prob_{str(infer_prob)}_angle_{gt_angle_flag}_ns_{num_samples}_pred2d_{pred_2d_flag}_{method}.json")
            
            file_write_meta = open(path_meta, 'w')
            #print("joints_3d_pred_gather", joints_3d_pred_repeat_gather.shape)
            if dist.get_rank() == 0:
                print("joints_3d_pred", joints_3d_pred_repeat_gather.shape)
                print("joints_3d_gt", joints_3d_gt_gather.shape)
                print("joints_3d_pred_samples", results_list.shape)
                print("ass_add", np.array(ass_add).shape)
                print("depth_path_lst", len(depth_path_lst))
                print("pose_gt_lst", np.array(pose_gt_gather).shape)
                if pred_2d_flag:
                    print("uv_pred_list", uv_pred_list.shape)
                    print("uv_gt_list", uv_gt_list.shape)
                    print("uv_pck_list", uv_pck_list.shape)
            
            meta_json["diff epoch"] = str(start_epoch).zfill(5)
            meta_json["sampler name"] = method
            meta_json["joints_3d_pred"] = joints_3d_pred_repeat_gather.detach().cpu().numpy().tolist()
            meta_json["joints_3d_gt"] = joints_3d_gt_gather.detach().cpu().numpy().tolist()
            #meta_json["joints_3d_pred_samples"] = results_list.detach().cpu().numpy().tolist()
            meta_json["ass_add"] = ass_add
            meta_json["ass_per_add"] = ass_per_add
            meta_json["pose_gt_lst"] = pose_gt_gather
            meta_json["depth_path_lst"] = depth_path_lst
            if pred_2d_flag:
                meta_json["uv_pred_list"] = uv_pred_list.tolist()
                meta_json["uv_gt_list"] = uv_gt_list.tolist()
                meta_json["uv_pck_list"] = uv_pck_list.tolist()
            json_save = json.dumps(meta_json, indent=1)
            file_write_meta.write(json_save)
            file_write_meta.close()
            
            if dist.get_rank() == 0:    
#                print("percentage", np.percentile(uv_pred_list, [i * 10 for i in range(1, 10)]))            
                with open(file_name, "w") as f:
                    print_to_screen_and_file(
                    f, "Analysis results for dataset: {}".format(split[mode][2])
                    )
                    print_to_screen_and_file(
                    f, "Number of frames in this dataset: {}".format(len(ass_add))
                    )
                    print_to_screen_and_file(f, "")
                    
                    # print add
                    print_to_screen_and_file(
                        f, " ADD AUC: {:.5f}".format(ass_add_results["add_auc"])
                    )
                    print_to_screen_and_file(
                        f,
                           " ADD  AUC threshold: {:.5f} m".format(ass_add_results["add_auc_thresh"]),
                    )
                    print_to_screen_and_file(
                        f, " ADD  Mean: {:.5f}".format(ass_add_results["add_mean"])
                    )
                    print_to_screen_and_file(
                        f, " ADD  Median: {:.5f}".format(ass_add_results["add_median"])
                    )
                    print_to_screen_and_file(
                        f, " ADD  Std Dev: {:.5f}".format(ass_add_results["add_std"]))
                    print_to_screen_and_file(f, "")
                    
                    print_to_screen_and_file(
                        f, f" uv_kps_pck : {np.mean(uv_kps_pck_list, axis=0)}")
                    
                    
                    # print mAP
                    for thresh, avg_map in zip(thresholds, ass_mAP_results):
                        print_to_screen_and_file(
                        f, " acc thresh: {:.5f} m".format(thresh)
                        )
                        print_to_screen_and_file(
                        f, " acc: {:.5f} %".format(float(avg_map))
                        )
                        ass_mAP_dict[str(thresh)] = float(avg_map)
                    print_to_screen_and_file(f, "")
                    
                    
                    # print mAP
                    for thresh, avg_map in zip(thresholds, kps_mAP_results):
                        print_to_screen_and_file(
                        f, " acc thresh: {:.5f} m".format(thresh)
                        )
                        print_to_screen_and_file(
                        f, " acc: {:.5f} %".format(float(avg_map))
                        )
                        ass_mAP_dict[str(thresh)] = float(avg_map)
                    print_to_screen_and_file(f, "")
                    
                    # print acc
                    for thresh, avg_acc in zip(acc_thresholds, angles_results):
                        print_to_screen_and_file(
                        f, " acc thresh: {:.5f} degree".format(thresh)
                        )
                        print_to_screen_and_file(
                        f, " acc: {:.5f} %".format(float(avg_acc))
                        )
                        angles_dict[str(thresh)] = float(avg_acc)
                    print_to_screen_and_file(f, "")
                    
                    # print add
                    print_to_screen_and_file(
                        f, " ADD AUC: {:.5f}".format(kps_add_results["add_auc"])
                    )
                    print_to_screen_and_file(
                        f,
                           " ADD  AUC threshold: {:.5f} m".format(kps_add_results["add_auc_thresh"]),
                    )
                    print_to_screen_and_file(
                        f, " ADD  Mean: {:.5f}".format(kps_add_results["add_mean"])
                    )
                    print_to_screen_and_file(
                        f, " ADD  Median: {:.5f}".format(kps_add_results["add_median"])
                    )
                    print_to_screen_and_file(
                        f, " ADD  Std Dev: {:.5f}".format(kps_add_results["add_std"]))
                    print_to_screen_and_file(f, "")
                    
                    # print pck
                    if pred_2d_flag:
                        print_to_screen_and_file(
                        f, " PCK AUC: {:.5f}".format(pck_results["l2_error_auc"])
                        )
                        print_to_screen_and_file(
                        f, " PCK MEAN: {:.5f}".format(pck_results["l2_error_mean_px"])
                        )
                        print_to_screen_and_file(
                        f, " PCK MEDIAN: {:.5f}".format(pck_results["l2_error_median_px"])
                        )
                        print_to_screen_and_file(
                        f, " PCK STD: {:.5f}".format(pck_results["l2_error_std_px"])
                        )


    
    



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg, args = update_config()
    main(cfg)
#    find_seq_data_in_dir("/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/")












