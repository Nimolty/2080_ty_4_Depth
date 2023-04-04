import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from tqdm import tqdm
import os

# spdh
from depth_c2rp.DifferentiableRenderer.Kaolin.Renderer import DiffPFDepthRenderer
from depth_c2rp.datasets.datasets_spdh import Depth_dataset
from depth_c2rp.build import build_spdh_model
from depth_c2rp.utils.spdh_utils import load_spdh_model
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.analysis import add_from_pose, add_metrics, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred
from depth_c2rp.utils.utils import load_camera_intrinsics, set_random_seed, exists_or_mkdir
from depth_c2rp.configs.config import update_config

def network_inference(model, cfg, epoch_id, device, mAP_thresh=[0.02, 0.11, 0.01], add_thresh=0.1,angles_thresh=[2.5, 30.0, 2.5]):
    # set eval mode
    #device = model.device
    model.eval()
    
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    model_cfg = cfg["MODEL"]
    testing_data_dir = dataset_cfg["TESTING_ROOT"]
    camera_K = load_camera_intrinsics(os.path.join(testing_data_dir, "_camera_settings.json"))
    dr_engine = cfg["DR"]["ENGINE"]
    eval_cfg = cfg["EVAL"]
    
    test_dataset = Depth_dataset(train_dataset_dir=testing_data_dir,
                                     val_dataset_dir=testing_data_dir,
                                     joint_names=[f"panda_joint_3n_{i+1}" for i in range(int(dataset_cfg["NUM_JOINTS"]) // 2)],
                                     run=[0],
                                     init_mode="train", 
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
                                   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_cfg["BATCH_SIZE"], shuffle=True, \
                                               num_workers=int(eval_cfg["NUM_WORKERS"]), pin_memory=True, drop_last=False)
    
    # Inference 
    ass_add = []
    ass_mAP = []
    angles_acc = []
    K = torch.from_numpy(camera_K).to(device)
    
    # Evaludation Thresholds
    start_thresh_mAP, end_thresh_mAP, interval_mAP = mAP_thresh
    thresholds = np.arange(start_thresh_mAP, end_thresh_mAP, interval_mAP)
    thresh_length = len(thresholds)
    start_angle_acc, end_angle_acc, interval_acc = angles_thresh
    acc_thresholds = np.arange(start_angle_acc, end_angle_acc, interval_acc)
    
    # Initializing DR Machine
    dr_iter_num = int(cfg["DR_ITER_NUM"])
    img_w, img_h = dataset_cfg["RAW_RESOLUTION"]
    img_size = (img_h, img_w)
    
    if dr_engine == "Kaolin":
        DPRenderer = DiffPFDepthRenderer(cfg, device)
        DPRenderer.load_mesh()
        DPRenderer.set_camera_intrinsics(K, width=img_w, height=img_h)
    
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
#            if batch_idx >= 3:
#                break
            joints_3d_gt = batch['joints_3D_Z'].to(device).float()
            heatmap_gt = batch['heatmap_25d'].to(device).float()
            input_K = batch['K_depth'].to(device).float()
            input_fx = batch['K_depth'].to(device).float()[:, 0, 0]
            input_fy = batch['K_depth'].to(device).float()[:, 1, 1]
            
            if model_cfg["INPUT_TYPE"] == "XYZ":
                input_tensor = batch['xyz_img'].to(device).float()
            else:
                raise ValueError
            
            outputs = model(input_tensor)
            b, c, h, w = heatmap_gt.size()
            
            if model_cfg["NAME"] == "stacked_hourglass":
                heatmap_pred = outputs['heatmaps']
            elif "dreamhourglass" in model_cfg["NAME"]:
                heatmap_pred = outputs[-1]
            else:
                raise ValueError
            
            heatmap_pred, joints_3d_pred = get_joint_3d_pred(heatmap_pred, cfg, h, w, c, input_K)
            #print("joints_3d_pred.shape", joints_3d_pred.shape)
            
            joints_pred, joints_gt = joints_3d_pred, joints_3d_gt.detach().cpu().numpy() # B x N x 3
            ass_add_mean = batch_add_from_pose(joints_pred, joints_gt)
            ass_add = ass_add + ass_add_mean
            
            ass_mAP_mean = batch_mAP_from_pose(joints_pred, joints_gt,thresholds) # 
            ass_mAP.append(ass_mAP_mean)
            
    
    ass_add_results = add_metrics(np.array(ass_add), add_thresh)
    ass_mAP_results = np.round(np.mean(ass_mAP, axis=0) * 100, 2)
    ass_mAP_dict = dict()
    
    # Print File and Save Results
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    results_path = os.path.join(save_path, "NUMERICAL_RESULTS")
    exists_or_mkdir(save_path)
    exists_or_mkdir(results_path)
    exp_id = cfg["EXP_ID"]
    file_name = os.path.join(results_path, f"EXP{str(exp_id).zfill(2)}_{str(epoch_id).zfill(3)}_{str(dr_iter_num)}.txt")
    
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
        
    
            
    return ass_add_results, ass_mAP_dict
            
            
if __name__ == "__main__":
    cfg, args = update_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    
    model = build_spdh_model(cfg)
    start_epoch, global_iter = 0, 0
    epoch_id = cfg["EPOCH_ID"]
    exp_id = cfg["EXP_ID"]
    model = model.to(device)
    network_inference(model, cfg, epoch_id, device)
        
    
    
    
















