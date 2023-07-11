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
import pytorch3d.transforms as transforms

# spdh
from depth_c2rp.utils.utils import load_camera_intrinsics, set_random_seed, exists_or_mkdir, visualize_inference_results
from depth_c2rp.configs.config import update_config
from depth_c2rp.utils.spdh_utils import reduce_mean, init_spdh_model, compute_kps_joints_loss
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred, log_and_visualize_single
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat
from depth_c2rp.utils.analysis import flat_add_from_pose, add_metrics, add_from_pose, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles, batch_outlier_removal_pose, batch_repeat_add_from_pose, batch_repeat_mAP_from_pose, batch_repeat_acc_from_joint_angles

from depth_c2rp.models.backbones.dream_hourglass import ResnetSimple, SpatialSoftArgmax
#from depth_c2rp.datasets.datasets_voxel_ours import Voxel_dataset
from depth_c2rp.datasets.datasets_diffusion_ours import Diff_dataset
#from depth_c2rp.voxel_utils.voxel_network import build_voxel_simple_network, load_simplenet_model

# diffusion
from depth_c2rp.diffusion_utils.diffusion_network import build_diffusion_network, build_simple_network, load_simplenet_model
from depth_c2rp.diffusion_utils.diffusion_losses import get_optimizer, optimization_manager
from depth_c2rp.diffusion_utils.diffusion_ema import ExponentialMovingAverage
from depth_c2rp.diffusion_utils.diffusion_utils import diff_save_weights, diff_load_weights
from depth_c2rp.diffusion_utils.diffusion_sampling import get_sampling_fn

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
    val_dataset_dir=dataset_cfg["VAL_ROOT"],
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
                                     sigma=dataset_cfg["SIGMA"],
                                     norm_type=dataset_cfg["NORM_TYPE"],
                                     network_input=model_cfg["INPUT_TYPE"],
                                     network_task=model_cfg["TASK"],
                                     depth_range=dataset_cfg["DEPTH_RANGE"],
                                     depth_range_type=dataset_cfg["DEPTH_RANGE_TYPE"],
                                     aug_type=dataset_cfg["AUG_TYPE"],
                                     aug_mode=False,
                                     change_intrinsic=dataset_cfg["CHANGE_INTRINSIC"],
                                     uv_input=False,
                                     cond_uv_std=train_cfg["COND_UV_STD"],
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
    results_path = os.path.join(save_path, "NUMERICAL_RESULTS")
    info_path = os.path.join(save_path, "INFERENCE_LOGS")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    exists_or_mkdir(results_path)
    exists_or_mkdir(info_path)
    
    # Inference 
    ass_add = []
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
                                                                                          
    heatmap_model = ResnetSimple(
                                n_keypoints=cfg["DATASET"]["NUM_JOINTS"] * 2,
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
    joint_dim = int(cfg["DIFF_MODEL"]["JOINT_DIM"])

    print("len_dataloader", len(training_loader))
    
#    with torch.no_grad():
    gt_angle_flag = train_cfg["GT_ANGLE_FLAG"]

    #split = {"Real" : [real_sampler, real_loader, real_dataset_dir], "Validation" : [val_sampler, val_loader, val_dataset_dir]}
    #split = {"Validation" : [val_sampler, val_loader, val_dataset_dir]}
    split = {"Real" : [real_sampler, real_loader, real_dataset_dir],}
    heatmap_model.eval()
    model.eval()
    simplenet_model.eval()
    time_list = []
    time_list2 = []
    meta_json = []
    uv_pred_list = []

    with torch.no_grad():
        for mode, value in split.items():
            sampler, loader = value[0], value[1]
            sampler.set_epoch(start_epoch)
            
            for batch_idx, batch in enumerate(tqdm(loader)):
                batch_json = {}
                curr_loss = 0.0
                
                #input_tensor = batch["xyz_img_scale"].float().to(device)
                #xyz_img = batch["xyz_img"].detach().cpu().numpy()
                joints_3d_gt = batch['joints_3D_Z'].to(device, non_blocking=True).float()
                joints_kps_3d_gt = batch["joints_3D_kps"].to(device, non_blocking=True).float()
                joints_1d_gt = batch["joints_7"].to(device).float()
                pose_gt = batch["R2C_Pose"].to(device).float()
                #depth_img_vis = batch["depthvis"].detach().cpu().numpy()
                #depth_path = batch["depth_path"]
                #batch_json["depth_path"] = depth_path[0]

                
                
#                torch.cuda.synchronize()
#                t1 = time.time()
#                input_tensor = batch["xyz_img_scale"].float().to(device)
#                _, _, h, w = input_tensor.shape
#                heatmap_pred = heatmap_model(input_tensor)[-1]
#                heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
#                _, c, _, _ = heatmap_pred.shape
#                
#                uv_pred = softargmax_uv(heatmap_pred[:, :c//2, :, :])   
#                
#                uv_pred_list += (((uv_pred / w - batch["joints_2D_uv"].cuda().float()) * w).abs().detach().cpu().numpy().reshape(uv_pred.shape[0], -1).tolist())
#     
#                batch["joints_2D_uv"] = uv_pred / w
                
                
                joints_2d = batch["joints_2D_uv"].float().to(device, non_blocking=True)
                bs, N, _ = joints_2d.shape
                joints_2d_yummy = torch.zeros(bs, N, 1).to(device, non_blocking=True)
                joints_2d = torch.cat([joints_2d, joints_2d_yummy], dim=-1).float()
                
                joints_2d_repeat = joints_2d.repeat(num_samples, 1, 1)

                        
                # Generate and save samples
                #ema.store(model.parameters())

                ema.copy_to(model.parameters())
                trajs, results = model.module.sampling_fn(
                            model.module.model,
                            condition=joints_2d_repeat
                        )  # [b ,j ,3]
                
                
                # # results: [b, j, 16], i.e., the end pose of each traj
                results = torch.from_numpy(results).float().to(device) # (bs x num_samples, j, 16)
                results = results.reshape(num_samples, bs, -1, joint_dim) 
                results = results.permute(1, 0, 2, 3) # bs x num_samples x j x 16
                results = torch.mean(results, dim=1) # bs x j x 16
                
                assert results.shape[1] == 1
                assert results.shape[0] == bs
                assert results.shape[2] == joint_dim
                results = results.squeeze(1)
                
                
                R2C_o6d_pred = results[:, :6] # bs x 6
                R2C_Trans_pred = results[:, 6:9][..., None] # bs x 3 x 1
                joints_angle_pred = results[:, 9:][..., None] # bs x 7 x 1
                R2C_Rot_pred = transforms.rotation_6d_to_matrix(R2C_o6d_pred) # bs x 3 x 3
                            
                if gt_angle_flag:
                    joints_angle_pred = joints_1d_gt[..., None]
                
                
                joints_kps_3d_pred = compute_kps_joints_loss(R2C_Rot_pred, R2C_Trans_pred, joints_angle_pred, device)      
                
                kps_add = kps_add + batch_add_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy())
                kps_mAP.append(batch_mAP_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy(), thresholds))

                angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred.detach().cpu().numpy(), joints_1d_gt.detach().cpu().numpy()[:, :, None], acc_thresholds)
                angles_acc.append(angles_acc_mean)
                
            torch.distributed.barrier()
            angles_acc = np.concatenate(angles_acc, axis=0)
            kps_mAP = np.concatenate(kps_mAP, axis=0)
            
            print(uv_pred_list)
#            uv_pred_list = distributed_concat(torch.from_numpy(np.array(uv_pred_list)).to(device), len(sampler.dataset))
            angles_acc = distributed_concat(torch.from_numpy(np.array(angles_acc)).to(device), len(sampler.dataset))
            kps_add  = distributed_concat(torch.from_numpy(np.array(kps_add)).to(device), len(sampler.dataset))
            kps_mAP = distributed_concat(torch.from_numpy(np.array(kps_mAP)).to(device), len(sampler.dataset))
#            
#            joints_3d_pred_gather = distributed_concat(torch.cat(joints_3d_pred_lst, dim=0), len(sampler.dataset))
#            joints_angle_pred_gather = distributed_concat(torch.cat(joints_angle_pred_lst, dim=0), len(sampler.dataset))
#            pose_pred_gather = distributed_concat(torch.cat(pose_pred_lst, dim=0), len(sampler.dataset))
#            joints_3d_gt_gather = distributed_concat(torch.cat(joints_3d_gt_lst, dim=0), len(sampler.dataset))
#            joints_angle_gt_gather = distributed_concat(torch.cat(joints_angle_gt_lst, dim=0), len(sampler.dataset))
#            pose_gt_gather = distributed_concat(torch.cat(pose_gt_lst, dim=0), len(sampler.dataset))
#            
#            kps_pred_gather = distributed_concat(torch.cat(kps_pred_lst, dim=0), len(sampler.dataset))
#            kps_gt_gather = distributed_concat(torch.cat(kps_gt_lst, dim=0), len(sampler.dataset))

            
            angles_acc = angles_acc.detach().cpu().numpy().tolist()
            kps_add = kps_add.detach().cpu().numpy().tolist()
            kps_mAP = kps_mAP.detach().cpu().numpy().tolist()
#            uv_pred_list = uv_pred_list.detach().cpu().numpy().reshape(-1)
            
            
            
            angles_results = np.round(np.mean(angles_acc,axis=0)*100, 2)
            angles_dict = dict()
            kps_add_results = add_metrics(kps_add, add_thresh)
            kps_mAP_results = np.round(np.mean(kps_mAP, axis=0) * 100, 2)
            kps_mAP_dict = dict()

            
            change_intrinsic_flag = dataset_cfg["CHANGE_INTRINSIC"]
            real_name = dataset_cfg["REAL_ROOT"].split('/')[-2]
            file_name = os.path.join(results_path, f"Epoch_{start_epoch}_{real_name}_repro_{change_intrinsic_flag}_angle_{gt_angle_flag}.txt")
#            path_meta = os.path.join(results_path, f"Epoch_{start_epoch}_{real_name}_repro_{change_intrinsic_flag}_angle_{gt_angle_flag}.json")
            
#            file_write_meta = open(path_meta, 'w')
#            json_save = json.dumps(meta_json, indent=1)
#            file_write_meta.write(json_save)
#            file_write_meta.close()
            
            if dist.get_rank() == 0:    
#                print("percentage", np.percentile(uv_pred_list, [i * 10 for i in range(1, 10)]))            
                with open(file_name, "w") as f:
                    print_to_screen_and_file(
                    f, "Analysis results for dataset: {}".format(split[mode][2])
                    )
                    print_to_screen_and_file(
                    f, "Number of frames in this dataset: {}".format(len(kps_add))
                    )
                    print_to_screen_and_file(f, "")
                    
                    
                    # print mAP
                    for thresh, avg_map in zip(thresholds, kps_mAP_results):
                        print_to_screen_and_file(
                        f, " acc thresh: {:.5f} m".format(thresh)
                        )
                        print_to_screen_and_file(
                        f, " acc: {:.5f} %".format(float(avg_map))
                        )
                        kps_mAP_dict[str(thresh)] = float(avg_map)
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
                


    
    



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg, args = update_config()
    main(cfg)
#    find_seq_data_in_dir("/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/")












