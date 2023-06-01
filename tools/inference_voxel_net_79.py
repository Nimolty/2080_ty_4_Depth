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


# spdh
from depth_c2rp.utils.utils import load_camera_intrinsics, set_random_seed, exists_or_mkdir, visualize_inference_results
from depth_c2rp.configs.config import update_config
from depth_c2rp.utils.spdh_utils import reduce_mean, init_spdh_model, compute_kps_joints_loss
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred, log_and_visualize_single
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat
from depth_c2rp.utils.analysis import flat_add_from_pose, add_metrics, add_from_pose, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles, batch_outlier_removal_pose

# voxel
from depth_c2rp.datasets.datasets_voxel_ours import Voxel_dataset
from depth_c2rp.voxel_utils.voxel_network import build_voxel_network, init_voxel_optimizer, load_voxel_model, save_weights, build_voxel_refine_network, load_refine_model, build_voxel_simple_network, load_simplenet_model
from depth_c2rp.voxel_utils.voxel_batch_utils import prepare_data, get_valid_points, get_occ_vox_bound, get_miss_ray, compute_ray_aabb, compute_gt, get_embedding_ours, get_pred, compute_loss, get_embedding, adapt_lr
from depth_c2rp.voxel_utils.refine_batch_utils import get_pred_refine, compute_refine_loss

from depth_c2rp.models.backbones.dream_hourglass import ResnetSimple, SpatialSoftArgmax

#from inference import network_inference
#from inference_spdh_multi import network_inference


def main(cfg, mAP_thresh=[0.02, 0.11, 0.01], add_thresh=0.06,angles_thresh=[2.5, 30.0, 2.5]):
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
    val_dataset_dir=dataset_cfg["VAL_ROOT"],
    real_dataset_dir=dataset_cfg["REAL_ROOT"],
    camera_K = load_camera_intrinsics(os.path.join(training_data_dir, "_camera_settings.json"))
    
    # Build training and validation set
    training_dataset = Voxel_dataset(train_dataset_dir=dataset_cfg["TRAINING_ROOT"],
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
                                     uv_input=cfg["voxel_network"]["uv_input"],
                                     )
                                    
    training_dataset.train()
    train_sampler = DistributedSampler(training_dataset)
    training_loader = DataLoader(training_dataset, sampler=train_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True) 
    
    
    val_dataset = copy.copy(training_dataset)  
    val_dataset.eval()    
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    real_dataset = copy.copy(training_dataset)
    real_dataset.real()
    real_sampler = DistributedSampler(real_dataset)
    real_loader = DataLoader(real_dataset, sampler=real_sampler, batch_size=train_cfg["BATCH_SIZE"],
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
    model = build_voxel_network(cfg, device)
    refine_model = build_voxel_refine_network(cfg,device)
    simplenet_model = build_voxel_simple_network(cfg)
    
    if cfg["RESUME"]:
        print("Resume Model Checkpoint")
        checkpoint_paths = os.listdir(checkpoint_path)
        checkpoint_paths.sort() 
        #this_ckpt_path = os.path.join(checkpoint_path, "model_068.pth")  
        this_ckpt_path = os.path.join(checkpoint_path, "model.pth")  
        print('this_ckpt_path', this_ckpt_path)
        model, stage_one_epoch, global_iter = load_voxel_model(model,this_ckpt_path, device)
        print("successfully loaded!")
    
    if cfg["RESUME_REFINE"]:
        print("Resume Model Checkpoint")
        checkpoint_paths = os.listdir(checkpoint_path)
        checkpoint_paths.sort() 
        this_ckpt_path = os.path.join(checkpoint_path, "refine_model.pth")  
        #this_ckpt_path = os.path.join(checkpoint_path, "refine_model_096.pth")  
        print('this_ckpt_path', this_ckpt_path)
        refine_model, stage_two_epoch, global_iter = load_refine_model(refine_model, this_ckpt_path, device)
        print("successfully loaded Refine Model!")

    start_epoch, global_iter = 0, 0
    model = model.to(device)
    refine_model = refine_model.to(device)
    simplenet_model = simplenet_model.to(device)
    num_gpus = torch.cuda.device_count()
    print("device", device)
    #if num_gpus > 1:
    print('use {} gpus!'.format(num_gpus))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    refine_model = torch.nn.parallel.DistributedDataParallel(refine_model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    simplenet_model = torch.nn.parallel.DistributedDataParallel(simplenet_model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
                                                
    #model = torch.compile(model, mode="reduce-overhead")
                                                
    heatmap_model = ResnetSimple(
                                n_keypoints=cfg["DATASET"]["NUM_JOINTS"] * 2,
                                full=cfg["voxel_network"]["full"],
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
    voxel_optimizer, refine_optimizer, scheduler = init_voxel_optimizer(model, refine_model, cfg)
    
    
    #heatmap_model = torch.compile(heatmap_model, mode="reduce-overhead")
    
    
    # Build Loss Function
    heatmap_criterion = torch.nn.MSELoss()
    woff_criterion = torch.nn.MSELoss()
    loss_cfg = cfg["LOSS"]
    

    print("len_dataloader", len(training_loader))
    
#    with torch.no_grad():
    epoch = stage_one_epoch
    gt_angle_flag = train_cfg["GT_ANGLE_FLAG"]

    split = {"Real" : [real_sampler, real_loader, real_dataset_dir], "Validation" : [val_sampler, val_loader, val_dataset_dir]}
    #split = {"Validation" : [val_sampler, val_loader, val_dataset_dir]}
    #split = {"Real" : [real_sampler, real_loader, real_dataset_dir],}
    heatmap_model.eval()
    model.eval()
    refine_model.eval()
    simplenet_model.eval()
    time_list = []
    time_list2 = []

    with torch.no_grad():
        for mode, value in split.items():
            sampler, loader = value[0], value[1]
            sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(tqdm(loader)):
                curr_loss = 0.0
                
                
                input_tensor = batch["xyz_img_scale"].float().to(device)
                joints_3d_gt = batch['joints_3D_Z'].to(device).float()
                joints_kps_3d_gt = batch["joints_3D_kps"].to(device).float()
                joints_1d_gt = batch["joints_7"].to(device).float()
                pose_gt = batch["R2C_Pose"].to(device).float()
                
                torch.cuda.synchronize()
                t1 = time.time()
                b, _, h, w = input_tensor.shape
                heatmap_pred = heatmap_model(input_tensor)[-1]
                heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
                _, c, _, _ = heatmap_pred.shape
                
                uv_pred = softargmax_uv(heatmap_pred[:, :c//2, :, :])          
                batch["joints_2D_uv"] = uv_pred
                
                
                loss_dict, data_dict = model(batch, "test", epoch-1)
                

                if cfg["TRAIN"]["FIRST_EPOCHS"] < epoch:
                    refine_forward_times = cfg["refine_voxel_network"]["refine_forward_times"]
                    if cfg["TRAIN"]["SECOND_EPOCHS"] < epoch:
                        refine_hard_neg = cfg["refine_voxel_network"]["refine_hard_neg"]
                        refine_hard_neg_ratio = cfg["refine_voxel_network"]["refine_hard_neg_ratio"]
                        loss_cfg["pos_coeff"] = cfg["refine_voxel_network"]["pos_coeff"]
                    else:
                        refine_hard_neg = False
                        refine_hard_neg_ratio = 0.0
                    for cur_iter in range(refine_forward_times):
                        loss_dict_refine, data_dict = refine_model(data_dict, loss_dict, cur_iter, refine_forward_times, epoch, "test", refine_hard_neg, refine_hard_neg_ratio)
                    
                    loss_dict = loss_dict_refine
                    
                
                
                joints_3d_pred = data_dict["pred_pos"].view(b, -1, 3)
                joints_angle_pred, pose_pred = simplenet_model(joints_3d_pred.clone(), joints_1d_gt[..., None], gt_angle_flag) 
                
                if gt_angle_flag:
                    joints_angle_pred = joints_1d_gt[..., None]
                
                joints_angle_pred_lst.append(joints_angle_pred)
                joints_angle_gt_lst.append(joints_1d_gt)
                joints_3d_pred_lst.append(joints_3d_pred)
                joints_3d_gt_lst.append(joints_3d_gt)
                pose_pred_lst.append(pose_pred)
                pose_gt_lst.append(pose_gt) 
                
                
                joints_kps_3d_pred = compute_kps_joints_loss(pose_pred[:, :3, :3], (pose_pred[:, :3, 3])[:, :, None], joints_angle_pred, device)  
                
                
                kps_add = kps_add + batch_add_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy())
                kps_mAP.append(batch_mAP_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy(),thresholds))
        
                
                kps_pred_lst.append(joints_kps_3d_pred)
                kps_gt_lst.append(joints_kps_3d_gt)
                
                # joints_pred, joints_gt = joints_3d_pred, joints_3d_gt.detach().cpu().numpy() # B x N x 3
                joints_pred, joints_gt = joints_3d_pred.detach().cpu().numpy(), joints_3d_gt.detach().cpu().numpy() # B x N x 3
                
                
                ass_add_mean = batch_add_from_pose(joints_pred, joints_gt)
                ass_add = ass_add + ass_add_mean
                
                ass_mAP_mean = batch_mAP_from_pose(joints_pred, joints_gt,thresholds) # 
                ass_mAP.append(ass_mAP_mean)
                
                angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred.detach().cpu().numpy(), joints_1d_gt.detach().cpu().numpy()[:, :, None], acc_thresholds)
                angles_acc.append(angles_acc_mean)
                
            torch.distributed.barrier()
            angles_acc = np.concatenate(angles_acc, axis=0)
            ass_mAP = np.concatenate(ass_mAP, axis=0)
            kps_mAP = np.concatenate(kps_mAP, axis=0)
            #print(angles_acc.shape)
            
            ass_add = distributed_concat(torch.from_numpy(np.array(ass_add)).to(device), len(sampler.dataset))
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

            
            ass_add = ass_add.detach().cpu().numpy().tolist()
            ass_mAP = ass_mAP.detach().cpu().numpy().tolist()
            angles_acc = angles_acc.detach().cpu().numpy().tolist()
            kps_add = kps_add.detach().cpu().numpy().tolist()
            kps_mAP = kps_mAP.detach().cpu().numpy().tolist()
            
            angles_results = np.round(np.mean(angles_acc,axis=0)*100, 2)
            angles_dict = dict()
            kps_add_results = add_metrics(kps_add, add_thresh)
            kps_mAP_results = np.round(np.mean(kps_mAP, axis=0) * 100, 2)
            kps_mAP_dict = dict()
            ass_add_results = add_metrics(ass_add, add_thresh)
            ass_mAP_results = np.round(np.mean(ass_mAP, axis=0) * 100, 2)
            ass_mAP_dict = dict()
            
            change_intrinsic_flag = dataset_cfg["CHANGE_INTRINSIC"]
            file_name = os.path.join(results_path, f"Epoch_{epoch}_{mode}_repro_{change_intrinsic_flag}_angle_{gt_angle_flag}.txt")
            
            if dist.get_rank() == 0:                
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
                


    
    



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg, args = update_config()
    main(cfg)
#    find_seq_data_in_dir("/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/")

















