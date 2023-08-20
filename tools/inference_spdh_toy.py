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
from depth_c2rp.datasets.datasets_toy import Depth_dataset
from depth_c2rp.build import build_toy_spdh_model, build_mode_spdh_model
from depth_c2rp.utils.spdh_utils import load_spdh_model
from depth_c2rp.spdh_optimizers import init_toy_optimizer
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
    print("testing_data_dir", testing_data_dir)
    camera_K = load_camera_intrinsics(os.path.join(testing_data_dir, "_camera_settings.json"))
    dr_engine = cfg["DR"]["ENGINE"]
    eval_cfg = cfg["EVAL"]
    
    test_dataset = Depth_dataset(train_dataset_dir=testing_data_dir,
                                     val_dataset_dir=testing_data_dir,
                                     joint_names=[f"panda_joint_3n_{i+1}" for i in range(int(dataset_cfg["NUM_JOINTS"]) // 2)],
                                     run=[0],
                                     init_mode="train", 
                                     three_d_norm=cfg["THREE_D_NORM"],
                                     three_d_noise_mu1=cfg["THREE_D_NOISE_MU1"],
                                     three_d_noise_mu2=cfg["THREE_D_NOISE_MU2"],
                                     three_d_noise_mu3=cfg["THREE_D_NOISE_MU3"],
                                     three_d_noise_std1=cfg["THREE_D_NOISE_STD1"], 
                                     three_d_noise_std2=cfg["THREE_D_NOISE_STD2"], 
                                     three_d_noise_std3=cfg["THREE_D_NOISE_STD3"], 
                                     three_d_random_drop=cfg["THREE_D_RANDOM_DROP"],
                                     kps_14_name=cfg["KPS_14_NAME"]
                                     )
    print("three_d_norm", cfg["THREE_D_NORM"])
    print("three_d_noise_mu1", cfg["THREE_D_NOISE_MU1"])
    print("three_d_noise_mu2", cfg["THREE_D_NOISE_MU2"])
    print("three_d_noise_mu3", cfg["THREE_D_NOISE_MU3"])
    print("three_d_noise_std1", cfg["THREE_D_NOISE_STD1"])
    print("three_d_noise_std2", cfg["THREE_D_NOISE_STD2"])
    print("three_d_noise_std3", cfg["THREE_D_NOISE_STD3"])
    print("three_d_random_drop", cfg["THREE_D_RANDOM_DROP"])
                                   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_cfg["BATCH_SIZE"], shuffle=True, \
                                               num_workers=int(eval_cfg["NUM_WORKERS"]), pin_memory=True, drop_last=False)
    
    # Inference 
    ass_add = []
    ass_mAP = []
    angles_acc = []
    K = torch.from_numpy(camera_K).to(device)
    
    # Evaludation Thresholds
    start_angle_acc, end_angle_acc, interval_acc = angles_thresh
    acc_thresholds = np.arange(start_angle_acc, end_angle_acc, interval_acc)
    
    # Initializing DR Machine
    dr_iter_num = int(cfg["DR_ITER_NUM"])
    img_w, img_h = dataset_cfg["RAW_RESOLUTION"]
    img_size = (img_h, img_w)
    loss = torch.zeros(7).to(device)
    
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
#            if batch_idx >= 3:
#                break
            joints_3d = batch['joints_3D_Z'].to(device).float()
            joints_1d_gt = batch["joints_7"].to(device).float()
                         
            joints_1d_pred = model(torch.flatten(joints_3d, 1))
            loss += torch.mean(torch.abs(joints_1d_pred - joints_1d_gt), dim=0)
            
            angles_acc_mean = batch_acc_from_joint_angles(joints_1d_pred[:, :, None].detach().cpu().numpy(), joints_1d_gt[:, :, None].detach().cpu().numpy(), acc_thresholds) # 
            #print("np.array", np.array(angles_acc_mean).shape)
            angles_acc.append(angles_acc_mean)
    
    print("loss",loss / len(test_loader))
    print("len angles acc", len(angles_acc))
    angles_acc = np.concatenate(angles_acc, axis=0)
    print((np.array(angles_acc)).shape)
    angles_results = np.round(np.mean(angles_acc,axis=0)*100, 2)
    print("angles_results", angles_results)
    angles_dict = dict()
    
    # Print File and Save Results
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    results_path = os.path.join(save_path, "NUMERICAL_RESULTS")
    exists_or_mkdir(save_path)
    exists_or_mkdir(results_path)
    exp_id = cfg["EXP_ID"]
    data_id = testing_data_dir.split('/')[-1]
    file_name = os.path.join(results_path, f"EXP{str(exp_id).zfill(2)}_{str(epoch_id).zfill(3)}_{str(dr_iter_num)}_{data_id}.txt")
    
    with open(file_name, "w") as f:
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
        
    
            
    return angles_dict
            
            
if __name__ == "__main__":
    cfg, args = update_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    dataset_cfg = cfg["DATASET"]
    
    kwargs = {"dim" : dataset_cfg["NUM_JOINTS"] // 2 * 3, "h1_dim" :  1024, "out_dim" : 7}
    model = build_mode_spdh_model(kwargs, cfg["TOY_NETWORK"])
    start_epoch, global_iter = 0, 0
    optimizer, scheduler = init_toy_optimizer(model, cfg)
    
    if cfg["MODEL_PATH"]:
        path = cfg["MODEL_PATH"]
        model, optimizer, scheduler, start_epoch, global_iter = load_spdh_model(model, optimizer, scheduler, path, device)
        print("path", path)
    
    if cfg["SYN_TEST"]:
        cfg["DATASET"]["TESTING_ROOT"] = cfg["SYN_TEST"]
    
    epoch_id = cfg["EPOCH_ID"]
    exp_id = cfg["EXP_ID"]
    model = model.to(device)
    network_inference(model, cfg, epoch_id, device)
        
    
    
    
















