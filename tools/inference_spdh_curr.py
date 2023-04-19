import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.nn.parallel import DataParallel as DP
from tqdm import tqdm
import os
import cv2

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# spdh
from depth_c2rp.DifferentiableRenderer.Kaolin.Renderer import DiffPFDepthRenderer
from depth_c2rp.datasets.datasets_spdh import Depth_dataset
from depth_c2rp.build import build_whole_spdh_model
from depth_c2rp.utils.spdh_utils import load_spdh_model, write_prediction_and_gt, load_prediction_and_gt, compute_kps_joints_loss
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.analysis import add_from_pose, add_metrics, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred
from depth_c2rp.utils.utils import load_camera_intrinsics, set_random_seed, exists_or_mkdir, matrix_to_quaternion
from depth_c2rp.configs.config import update_config
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat

def network_inference(model, cfg, epoch_id, device, mAP_thresh=[0.02, 0.11, 0.01], add_thresh=0.06,angles_thresh=[2.5, 30.0, 2.5]):
    # set eval mode
    #device = model.device
    model.eval()
    
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    model_cfg = cfg["MODEL"]
    testing_data_dir = dataset_cfg["TESTING_ROOT"]
    #camera_K = load_camera_intrinsics(os.path.join(testing_data_dir, "_camera_settings.json"))
    dr_engine = cfg["DR"]["ENGINE"]
    eval_cfg = cfg["EVAL"]

    
    # Inference 
    ass_add = []
    ass_mAP = []
    angles_acc = []
    kps_add = []
    kps_mAP = []
    
    # lst
    joints_angle_pred_lst = []
    joints_3d_pred_lst = []
    pose_pred_lst = []
    joints_angle_gt_lst = []
    joints_3d_gt_lst = []
    pose_gt_lst = []
    
    
    #K = torch.from_numpy(camera_K).to(device)
    # Initializing DR Machine
    dr_iter_num = int(cfg["DR_ITER_NUM"])
    
    # Evaludation Thresholds
    start_thresh_mAP, end_thresh_mAP, interval_mAP = mAP_thresh
    thresholds = np.arange(start_thresh_mAP, end_thresh_mAP, interval_mAP)
    thresh_length = len(thresholds)
    start_angle_acc, end_angle_acc, interval_acc = angles_thresh
    acc_thresholds = np.arange(start_angle_acc, end_angle_acc, interval_acc)
    
    path_meta = cfg["LOAD_CURRENT_PREDGT"]
    kwargs = load_prediction_and_gt(path_meta)
    for key, value in kwargs.items():
        print("key", key)
        print("value.shape", value.shape)
    
    joints_3d_pred_gather = kwargs["joints_3d_pred_gather"]
    joints_3d_gt_gather = kwargs["joints_3d_gt_gather"]
    pose_pred_gather = kwargs["pose_pred_gather"]
    pose_gt_gather = kwargs["pose_gt_gather"]
    joints_angle_pred_gather =  kwargs["joints_angle_pred_gather"]
    joints_angle_gt_gather = kwargs["joints_angle_gt_gather"]
    kps_pred_gather = kwargs["kps_pred_gather"]
    kps_gt_gather = kwargs["kps_gt_gather"]
    
    ass_add_mean = batch_add_from_pose(joints_3d_pred_gather, joints_3d_gt_gather)
    ass_add = ass_add + ass_add_mean
    ass_mAP_mean = batch_mAP_from_pose(joints_3d_pred_gather, joints_3d_gt_gather,thresholds) # 
    ass_mAP.append(ass_mAP_mean)
    angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred_gather[:, :, None], joints_angle_gt_gather[:, :, None], acc_thresholds)
    angles_acc.append(angles_acc_mean)
    
    kps_add = kps_add + batch_add_from_pose(kps_pred_gather, kps_gt_gather)
    kps_mAP.append(batch_mAP_from_pose(kps_pred_gather, kps_gt_gathe,thresholds))
'    
    l1_mean = np.mean(np.abs(joints_angle_pred_gather - joints_angle_gt_gather), axis=0)
    print("l1_dis", l1_mean)
    l1_median = np.median(np.abs(joints_angle_pred_gather - joints_angle_gt_gather), axis=0)
    print("l1_dis", l1_median)
    
    ass_add_results = add_metrics(np.array(ass_add), add_thresh)
    ass_mAP_results = np.round(np.mean(ass_mAP, axis=0) * 100, 2)
    ass_mAP_dict = dict()
    angles_results = np.round(np.mean(angles_acc,axis=0)*100, 2)
    angles_dict = dict()
    
    kps_add_results = add_metrics(np.array(kps_add), add_thresh)
    kps_mAP_results = np.round(np.mean(kps_mAP, axis=0) * 100, 2)
    kps_mAP_dict = dict()
    
    # Print File and Save Results
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    results_path = os.path.join(save_path, "NUMERICAL_RESULTS")
    info_path = os.path.join(save_path, "INFERENCE_LOGS")
    exists_or_mkdir(save_path)
    exists_or_mkdir(results_path)
    exists_or_mkdir(info_path )
    exp_id = cfg["EXP_ID"]
    file_name = os.path.join(results_path, f"EXP{str(exp_id).zfill(2)}_{str(epoch_id).zfill(3)}_{str(dr_iter_num)}.txt")
    prediction_gt_name = os.path.join(info_path, f"EXP{str(exp_id).zfill(2)}_{str(epoch_id).zfill(3)}_{str(dr_iter_num)}.json")
    
   
    with open(file_name, "w") as f:
        print_to_screen_and_file(
        f, "Analysis results for dataset: {}".format(testing_data_dir)
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
        
    
            
    return ass_add_results, ass_mAP_dict, angles_dict
            
            
if __name__ == "__main__":
    cfg, args = update_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_whole_spdh_model(cfg, device)
    model = model.to(device)
    num_gpus = torch.cuda.device_count()
    print("device", device)
    model = torch.nn.DataParallel(model, device_ids=[0])
    
    
    start_epoch, global_iter = 0, 0
    optimizer, scheduler = init_optimizer(model, cfg)
    epoch_id = cfg["EPOCH_ID"]
    exp_id = cfg["EXP_ID"]
    
    if cfg["MODEL_PATH"]:
        path = cfg["MODEL_PATH"]
        model, optimizer, scheduler, start_epoch, global_iter = load_spdh_model(model, optimizer, scheduler, path, device)
        print("path", path)
        
    #model = model.to(device)
    network_inference(model, cfg, epoch_id, device)
        
    
    
    
















