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
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
import cv2

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# spdh
from depth_c2rp.DifferentiableRenderer.Kaolin.Renderer import DiffPFDepthRenderer
#from depth_c2rp.datasets.datasets_spdh_ours import Depth_dataset
from depth_c2rp.datasets.datasets_spdh import Depth_dataset
from depth_c2rp.build import build_whole_spdh_model
from depth_c2rp.utils.spdh_utils import load_spdh_model, write_prediction_and_gt, compute_kps_joints_loss
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.analysis import add_from_pose, add_metrics, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles, batch_outlier_removal_pose
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
                                     aug_mode=False,
                                     load_mask=True,
                                     mask_dict=dataset_cfg["MASK_DICT"],
                                     unnorm_depth=True,
                                     )
    #test_dataset.test()
    
    #test_sampler = DistributedSampler(test_dataset)
    
    #test_sampler = SequentialDistributedSampler(test_dataset, batch_size=eval_cfg["BATCH_SIZE"])
    test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=eval_cfg["BATCH_SIZE"], \
                                               num_workers=int(eval_cfg["NUM_WORKERS"]), pin_memory=True, drop_last=False)

    
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
    kps_pred_lst = []
    joints_angle_gt_lst = []
    joints_3d_gt_lst = []
    pose_gt_lst = []
    kps_gt_lst = []
    
    
    
    #K = torch.from_numpy(camera_K).to(device)
    
    # Evaludation Thresholds
    start_thresh_mAP, end_thresh_mAP, interval_mAP = mAP_thresh
    thresholds = np.arange(start_thresh_mAP, end_thresh_mAP, interval_mAP)
    thresh_length = len(thresholds)
    start_angle_acc, end_angle_acc, interval_acc = angles_thresh
    acc_thresholds = np.arange(start_angle_acc, end_angle_acc, interval_acc)
    
    
    # Initializing DR Machine
    dr_iter_num = int(cfg["DR_ITER_NUM"])
    if cfg["LINK_IDX"] == "whole":
        link_idx = cfg["LINK_IDX"]
    else:
        link_idx = int(cfg["LINK_IDX"])
    
    
    for batch_idx, batch in enumerate(tqdm(test_loader)):
#        if batch_idx >= 10:
#            break
        with torch.no_grad():
            joints_3d_gt = batch['joints_3D_Z'].to(device).float()
            joints_1d_gt = batch["joints_7"].to(device).float()
            joints_2d_depth = batch["joints_2D_depth"].to(device).float()
            heatmap_gt = batch['heatmap_25d'].to(device).float()
            pose_gt = batch["R2C_Pose"].to(device).float()
            input_K = batch['K_depth'].to(device).float()
            input_fx = batch['K_depth'].to(device).float()[:, 0, 0]
            input_fy = batch['K_depth'].to(device).float()[:, 1, 1]
            
            if model_cfg["INPUT_TYPE"] == "XYZ":
                input_tensor = batch['xyz_img'].to(device).float()
                #print("XYZ", input_tensor.shape)
            else:
                raise ValueError
            
            
            
            b, c, h, w = heatmap_gt.size()
            cam_params = {"h" : h, "w" : w, "c" : c, "input_K" : input_K}
            
#            for i in range(1000):
            torch.cuda.synchronize()
            t1 = time.time()
            heatmap_pred, joints_angle_pred, pose_pred, joints_3d_rob_pred, joints_3d_pred_norm, joints_3d_pred_tensor = model(input_tensor, cam_params)
            torch.cuda.synchronize()
            t2 = time.time()
            #print("time seq", t2 - t1)
            
            
            
        if dr_engine == "Kaolin" and batch_idx == 0:
                #print("input_K.shape", input_K.shape)
            DPRenderer = DiffPFDepthRenderer(cfg, device)
            DPRenderer.load_mesh()
            #print(input_K)
            K = np.array([[input_K[0, 0, 0].detach().cpu().numpy(), 0.0, input_K[0, 0, 2].detach().cpu().numpy()],
                           [0.0, input_K[0, 1, 1].detach().cpu().numpy(), dataset_cfg["INPUT_RESOLUTION"][1] // 2],
                           [0.0, 0.0, 1.0],
                          ])
            DPRenderer.set_camera_intrinsics(torch.from_numpy(K).to(device).float(), width=dataset_cfg["INPUT_RESOLUTION"][0], height=dataset_cfg["INPUT_RESOLUTION"][1])
            
        # Add Differentiable Renderer Kaolin
        # joints_angle_pred : B x 7 x 1
        # pose_pred : B x 3 x 4
        exists_or_mkdir(f"./check_blended_link_{link_idx}")
        if dr_engine == "Kaolin" and dr_iter_num > 0:
            #print("pose_pred", pose_pred)
            #print("pose_gt", pose_gt)
            batch_dt_quaternion = matrix_to_quaternion(pose_pred[:, :, :3])
            batch_dt_trans = pose_pred[:, :, 3]
            batch_dt_joints_pos = joints_angle_pred
            print(batch["rgb_path"])
            
            batch_dt_trans.requires_grad = True
            batch_dt_quaternion.requires_grad = True
            batch_dt_joints_pos.requires_grad = True
            
            DPRenderer.set_optimizer(batch_dt_quaternion, batch_dt_trans, batch_dt_joints_pos)
            DPRenderer.batch_mesh(b)
            
            batch_gt_simdepth = batch['unnorm_depth'].to(device).float()
            dr_w = dataset_cfg["INPUT_RESOLUTION"][0]
            dr_h = dr_w // 2
            start_edge, end_edge = (dr_w - dr_h) // 2, (dr_w + dr_h) // 2
            batch_gt_simdepth = batch_gt_simdepth[:, :, start_edge : end_edge, :]
            #print("max", torch.max(batch_gt_simdepth))
            #print("batch_gt_simdepth", batch_gt_simdepth.shape)
            batch_gt_simdepth = batch_gt_simdepth.permute(0, 2, 3, 1)
            #cv2.imwrite(f"./depth.png", batch_gt_simdepth[0][0][:, :, None].detach().cpu().numpy() * 255)
            
            
            all_res = np.zeros((b, 3, dr_h, dr_iter_num * dr_w))
            batch_gt_mask = batch["mask"].to(device).float()
            
            for update_idx in range(dr_iter_num):
                DPRenderer.GA_optimizer_zero_grad()
                DPRenderer.RT_optimizer_zero_grad()
                
                DPRenderer.concat_mesh()
                DPRenderer.Rasterize()
                res = DPRenderer.loss_forward(batch_gt_simdepth, batch_gt_mask,img_path=batch["rgb_path"], update_idx=update_idx, link_idx=link_idx)
                all_res[:, :, :, update_idx *dr_w : (update_idx+1) * dr_w] = res.transpose(0, 3, 1, 2)
                DPRenderer.loss_backward()
                
                batch_acc_from_joint_angles(joints_angle_pred[:, link_idx-1:link_idx, None].detach().cpu().numpy(), joints_1d_gt[:, link_idx-1:link_idx, None].detach().cpu().numpy(), acc_thresholds)
                
                DPRenderer.RT_optimizer_step()
                DPRenderer.GA_optimizer_step()
            
            grid_image = make_grid(torch.from_numpy(all_res), 1, normalize=False, scale_each=False) 
##        print("grid_image.shape", grid_image.shape) 
##        grid_image = PILImage.fromarray((grid_image.detach().cpu().numpy()))
##        grid_image.save(os.path.join(save_path, f"blend.png"))
##        grid_image = grid_image.detach().cpu().numpy()
##        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"./check_blended_link_{link_idx}/blend_cuda{device}_{str(batch_idx).zfill(4)}.png", grid_image.detach().cpu().numpy().transpose(1,2,0)[:,:,::-1])
            
            
        heatmap_pred, joints_3d_pred = get_joint_3d_pred(heatmap_pred, cfg, h, w, c, input_K)
        joints_angle_pred_lst.append(joints_angle_pred)
        pose_pred_lst.append(pose_pred)
        joints_3d_pred_lst.append(torch.from_numpy(joints_3d_pred).to(device))
        joints_angle_gt_lst.append(joints_1d_gt)
        joints_3d_gt_lst.append(joints_3d_gt)
        pose_gt_lst.append(pose_gt)
        
        
        # 
        joints_kps_3d_gt = batch["joints_3D_kps"].to(device).float()
        #print("joints_angle_pred.shape", joints_angle_pred.shape)
        #print("pose_pred", pose_pred.shape)
        
        #pose_pred = batch_outlier_removal_pose(joints_2d_depth, joints_3d_rob_pred, joints_3d_pred_norm, joints_3d_pred_tensor, (h, w))
        
        
        joints_kps_3d_pred = compute_kps_joints_loss(pose_pred[:, :3, :3], pose_pred[:, :3, 3][:, :, None], joints_angle_pred, device)
        kps_add = kps_add + batch_add_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy())
        kps_mAP.append(batch_mAP_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy(),thresholds))
        #print("joints_3d_pred.shape", joints_3d_pred.shape)
        
        kps_pred_lst.append(joints_kps_3d_pred)
        kps_gt_lst.append(joints_kps_3d_gt)
        
        joints_pred, joints_gt = joints_3d_pred, joints_3d_gt.detach().cpu().numpy() # B x N x 3
        ass_add_mean = batch_add_from_pose(joints_pred, joints_gt)
        ass_add = ass_add + ass_add_mean
        
        ass_mAP_mean = batch_mAP_from_pose(joints_pred, joints_gt,thresholds) # 
        ass_mAP.append(ass_mAP_mean)
        
        if link_idx == "whole":
            angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred.detach().cpu().numpy(), joints_1d_gt.detach().cpu().numpy()[:, :, None], acc_thresholds)
        else:
            angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred[:, link_idx-1:link_idx,:].detach().cpu().numpy(), joints_1d_gt[:, link_idx-1:link_idx, None].detach().cpu().numpy(), acc_thresholds)
        angles_acc.append(angles_acc_mean)
    
    
    
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
    
#    if dist.get_rank() == 0:
#        write_prediction_and_gt(prediction_gt_name, joints_3d_pred_gather, joints_3d_gt_gather, 
#                                         pose_pred_gather, pose_gt_gather, 
#                                         joints_angle_pred_gather, joints_angle_gt_gather,
#                                         kps_pred_gather, kps_gt_gather
#                                          )
        
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
    num_gpus = torch.cuda.device_count()
    print(num_gpus)
#    torch.multiprocessing.set_start_method('spawn')
#    if cfg["LOCAL_RANK"] != -1:
#        torch.cuda.set_device(cfg["LOCAL_RANK"])
#        device=torch.device("cuda",cfg["LOCAL_RANK"])
#        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_whole_spdh_model(cfg, device).to(device)
    model = torch.nn.DataParallel(model)
    #model = model.to(device)
    epoch_id = cfg["EPOCH_ID"]
    exp_id = cfg["EXP_ID"]
    print("device", device)
    #if num_gpus > 1:
#    print('use {} gpus!'.format(num_gpus))
#    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
#                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
#    
#    
    start_epoch, global_iter = 0, 0
    optimizer, scheduler = init_optimizer(model, cfg)
#    
#    
    if cfg["MODEL_PATH"]:
        path = cfg["MODEL_PATH"]
        model, optimizer, scheduler, start_epoch, global_iter = load_spdh_model(model, optimizer, scheduler, path, device)
        print("path", path)
        
    #model = model.to(device)
    network_inference(model, cfg, epoch_id, device)
        
    
    
    
















