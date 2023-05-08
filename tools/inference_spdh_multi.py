import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import math

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
from depth_c2rp.utils.spdh_utils import load_spdh_model, write_prediction_and_gt, compute_kps_joints_loss, compute_3d_error, compute_3n_loss_42_cam, depthmap2points, get_peak_pts
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.analysis import add_from_pose, add_metrics, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles, batch_outlier_removal_pose
from depth_c2rp.utils.analysis import batch_pck_from_pose, batch_1d_pck_from_pose, pck_metrics
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred
from depth_c2rp.utils.utils import load_camera_intrinsics, set_random_seed, exists_or_mkdir, matrix_to_quaternion, quaternion_to_matrix
from depth_c2rp.configs.config import update_config
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat
from depth_c2rp.utils.spdh_sac_utils import compute_rede_rt

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
    dr_loss = cfg["DR_LOSS"]
    dr_input = cfg["DR_INPUT"]
    dr_order = cfg["DR_ORDER"]
    cx_delta = cfg["CX_DELTA"]
    cy_delta = cfg["CY_DELTA"]
    
    print("dr_loss", dr_loss)
    print("dr_input", dr_input)
    print("dr_order", dr_order)
    
    
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
                                     cx_delta=cfg["CX_DELTA"],
                                     cy_delta=cfg["CY_DELTA"],
                                     )
    #test_dataset.test()
    
    #test_sampler = DistributedSampler(test_dataset)
    
    test_sampler = SequentialDistributedSampler(test_dataset, batch_size=eval_cfg["BATCH_SIZE"])
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_cfg["BATCH_SIZE"], \
                                               num_workers=int(eval_cfg["NUM_WORKERS"]), pin_memory=True, drop_last=False)

    
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
        
    # Print File and Save Results
    
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    results_path = os.path.join(save_path, "NUMERICAL_RESULTS", f"link_{link_idx}")
    info_path = os.path.join(save_path, "INFERENCE_LOGS")
    exists_or_mkdir(save_path)
    exists_or_mkdir(results_path)
    exists_or_mkdir(info_path )
    
    visual_lst = (np.linspace(0, len(test_loader), 30).astype(np.int64)).tolist()
    
    for batch_idx, batch in enumerate(tqdm(test_loader)):
#        if batch_idx >= 30:
#            break
        with torch.no_grad():
            joints_3d_gt = batch['joints_3D_Z'].to(device).float()
            joints_1d_gt = batch["joints_7"].to(device).float()
            joints_2d_depth = batch["joints_2D_depth"].to(device).float()
            joints_2d_dz = batch["joints_2d_dz"]
            heatmap_gt = batch['heatmap_25d'].to(device).float()
            pose_gt = batch["R2C_Pose"].to(device).float()
            input_K = batch['K_depth'].to(device).float()
            input_fx = batch['K_depth'].to(device).float()[:, 0, 0]
            input_fy = batch['K_depth'].to(device).float()[:, 1, 1]
            
            print(batch["depth_path"])
            
            if model_cfg["INPUT_TYPE"] == "XYZ":
                input_tensor = batch['xyz_img'].to(device).float()
                #print("XYZ", input_tensor.shape)
            else:
                raise ValueError
            
            
            
            b, c, h, w = heatmap_gt.size()
            cam_params = {"h" : h, "w" : w, "c" : c, "input_K" : input_K}
            #print(cam_params)
            
#            for i in range(1000):
            t1 = time.time()
                
            heatmap_pred, joints_angle_pred, pose_pred, joints_3d_rob_pred, joints_3d_pred_norm, joints_3d_pred_tensor, uv_pred, z_pred = model(input_tensor, cam_params, joints_1d_gt[..., None])
            heatmap_pred, joints_3d_pred = get_joint_3d_pred(heatmap_pred, cfg, h, w, c, input_K)
            
#            if batch_idx < 3: 
#                for c_idx in range(heatmap_gt.shape[1]):
#                    this_hm_gt = (heatmap_gt[0][c_idx:c_idx+1, :, :]).permute(1, 2, 0).detach().cpu().numpy()
#                    this_hm_pred = (heatmap_pred[0][c_idx:c_idx+1, :, :]).permute(1, 2, 0).detach().cpu().numpy()
#                    cv2.imwrite(f"./check/id_{batch_idx}_c{c_idx}_cx_{cx_delta}cy_{cy_delta}_gt.png", this_hm_gt *255) 
#                    cv2.imwrite(f"./check/id_{batch_idx}_c{c_idx}_cx_{cx_delta}cy_{cy_delta}_pred.png", this_hm_pred *255)
            if dist.get_rank() == 0:
                vis_xyz = batch["xyz_img_raw"].to(device).float()
                peaks_xyz = get_peak_pts(heatmap_pred[:, :14, :, :], vis_xyz, threshold=0.1) 
                for b_idx in range(peaks_xyz.shape[0]): 
                    vis_xyz_np = vis_xyz[b_idx].permute(1,2,0).reshape(-1, 3).detach().cpu().numpy()
                    peaks_xyz_np = peaks_xyz[b_idx].reshape(-1, 3).detach().cpu().numpy()
                    #print("num vertices of all", len(np.where(np.abs(vis_xyz_np[:, 2] - 1.5) < 1.5)[0]))
                    print("max z", np.max(vis_xyz_np[:,2])) 
                    print("valid num vertices of all", len(np.where(np.abs(vis_xyz_np[:, 2] - 1.5) < 1.5)[0]))
                    print("valid num vertices of peaks", len(np.where(peaks_xyz_np[:, 2] > 1e-3)[0])) 
                    print(peaks_xyz.shape)
                    
                    if batch_idx < 6:
                        np.savetxt(f"./pts/id_{batch_idx}_b{b_idx}_peaks.txt", peaks_xyz_np)   
                        np.savetxt(f"./pts/id_{batch_idx}_b{b_idx}_vis.txt", vis_xyz_np) 
             
            
            
            joints_kps_3d_gt = batch["joints_3D_kps"].to(device).float()
            depth_paths = batch["depth_path"]
            
            
            #print("joints_2d_depth", joints_2d_depth)
            #print("uv_pred", uv_pred)
            #print("joints_2d_dz", joints_2d_dz)
            #print("z_pred", z_pred)
            
            this_uv_pck = batch_pck_from_pose(joints_2d_depth.detach().cpu().numpy(), uv_pred.detach().cpu().numpy())
            this_z_pck = batch_1d_pck_from_pose(joints_2d_dz.detach().cpu().numpy(), z_pred.detach().cpu().numpy())
            
            pred_z.append(z_pred.detach().cpu().numpy().reshape(-1).tolist())
            gt_z.append(joints_2d_dz.detach().cpu().numpy().reshape(-1).tolist())
            uv_pck = uv_pck + this_uv_pck
            z_pck = z_pck + this_z_pck
            
            
            
#            if dist.get_rank() == 0 and (batch_idx in visual_lst):
#                for b_idx in range(joints_kps_3d_gt.shape[0]):
#                    depth_path = depth_paths[b_idx] 
#                    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0]
#                    pts = depthmap2points(depth_img, fx=502.30, fy=502.30, cx=319.5, cy=179.5)
#                    pts = pts.reshape(-1, 3)
#                    
#                    joints_3d_gt_np = joints_3d_gt.clone().detach().cpu().numpy()
#                    joints_3d_pred_np = joints_3d_pred_tensor.clone().detach().cpu().numpy()
#                    np.savetxt(f"./simpts/{str(batch_idx)}_{str(b_idx).zfill(2)}_sim.txt", pts)
#                    np.savetxt(f"./simpts/{str(batch_idx)}_{str(b_idx).zfill(2)}_kps_pred.txt", joints_3d_pred_np[b_idx])
#                    np.savetxt(f"./simpts/{str(batch_idx)}_{str(b_idx).zfill(2)}_kps_gt.txt", joints_3d_gt_np[b_idx])
            
            
            
#            print("time seq", time.time() - t1)
            
        if dr_engine == "Kaolin" and batch_idx == 0:
            DPRenderer = DiffPFDepthRenderer(cfg, device)
            DPRenderer.load_mesh()
            #print(input_K)
            K = np.array([[input_K[0, 0, 0].detach().cpu().numpy(), 0.0, input_K[0, 0, 2].detach().cpu().numpy()],
                           [0.0, input_K[0, 1, 1].detach().cpu().numpy(),  input_K[0, 1, 2].detach().cpu().numpy() - (dataset_cfg["INPUT_RESOLUTION"][0] - dataset_cfg["INPUT_RESOLUTION"][1]) // 2],
                           [0.0, 0.0, 1.0],
                          ])
                          
            #print("K", K)
            
            DPRenderer.set_camera_intrinsics(torch.from_numpy(K).to(device).float(), width=dataset_cfg["INPUT_RESOLUTION"][0], height=dataset_cfg["INPUT_RESOLUTION"][1])
             
        # Add Differentiable Renderer Kaolin
        # joints_angle_pred : B x 7 x 1
        # pose_pred : B x 4 x 4
        blended_path = os.path.join(results_path, f"{testing_data_dir.split('/')[-2]}_link_{link_idx}/dr_iter_{dr_iter_num}_{dr_order}")
        exists_or_mkdir(blended_path)
        if dr_engine == "Kaolin" and dr_iter_num > 0:
            batch_dt_quaternion = matrix_to_quaternion(pose_pred[:, :3, :3])
            batch_dt_trans = pose_pred[:, :3, 3]
            pose_pred_clone = pose_pred.clone()
            #print("batch_dt_quaternion_before", batch_dt_quaternion)
            #batch_dt_joints_pos = joints_angle_pred

                       
            #batch_dt_trans.requires_grad = True
            #batch_dt_quaternion.requires_grad = True
#            batch_dt_joints_pos.requires_grad = True
            
            
            
            #DPRenderer.set_all_optimizer(joints_angle_pred, batch_dt_quaternion, batch_dt_trans, dr_order)
            batch_gt_quaternion = matrix_to_quaternion(pose_gt[:, :3, :3])
            batch_gt_trans = pose_gt[:, :3, 3]
            
            DPRenderer.set_all_optimizer(joints_1d_gt[:, :, None], batch_gt_quaternion, batch_gt_trans, dr_order)
            
            
            DPRenderer.batch_mesh(b)
            
#            batch_gt_simdepth = batch['unnorm_depth'].to(device).float()
#            dr_w = dataset_cfg["INPUT_RESOLUTION"][0]
#            dr_h = dr_w // 2
#            start_edge, end_edge = (dr_w - dr_h) // 2, (dr_w + dr_h) // 2
#            batch_gt_simdepth = batch_gt_simdepth[:, :, start_edge : end_edge, :]
#            batch_gt_simdepth = batch_gt_simdepth.permute(0, 2, 3, 1)
#            
#            batch_gt_pfdepth = batch['unnorm_pf_depth'].to(device).float()
#            dr_w = dataset_cfg["INPUT_RESOLUTION"][0]
#            dr_h = dr_w // 2
#            start_edge, end_edge = (dr_w - dr_h) // 2, (dr_w + dr_h) // 2
#            batch_gt_pfdepth = batch_gt_pfdepth[:, :, start_edge : end_edge, :]
#            batch_gt_pfdepth = batch_gt_pfdepth.permute(0, 2, 3, 1)
#            
#            batch_gt_simxyz = batch["unnorm_xyz"].to(device) # B x H x W x 3
#            batch_gt_simxyz = batch_gt_simxyz[:, start_edge : end_edge, :, :]
#            batch_gt_pfxyz = batch["unnorm_pf_xyz"].to(device) # B x H x W x 3
#            batch_gt_pfxyz = batch_gt_pfxyz[:, start_edge : end_edge, :, :]
#            
#
#            all_res = np.zeros((b, 3, dr_h, dr_iter_num * dr_w))
#            batch_gt_mask = batch["mask"].to(device).float()
            #print("pose_pred", pose_pred)
            
#            batch_acc_from_joint_angles(joints_angle_pred[:, link_idx-1:link_idx, :].detach().cpu().numpy(), joints_1d_gt[:, link_idx-1:link_idx, None].detach().cpu().numpy(), acc_thresholds)
            
            for update_idx in range(dr_iter_num):
                optimizer_list = DPRenderer.GA_joint_dict[link_idx]
                for optimizer in optimizer_list:
                    optimizer.zero_grad()
                
                DPRenderer.concat_mesh()
                render_depths = DPRenderer.Rasterize()
                reconstruct_pts = compute_3n_loss_42_cam(pose_pred[:, :3, :3], (pose_pred[:, :3, 3])[:, :, None], joints_angle_pred, device)
                
#                if dist.get_rank() == 0 and (batch_idx in visual_lst):
#                    for b_idx in range(render_depths.shape[0]):
#                        this_render_depth = render_depths[b_idx]
#                        this_render_depth = (this_render_depth.reshape(-1, 3).detach().cpu().numpy()) * np.array([[1.0, -1.0, -1.0]])
#                        reconstruct_pt = reconstruct_pts[b_idx].detach().cpu().numpy()
#                        np.savetxt(f"./simpts/{str(batch_idx)}_{str(b_idx).zfill(2)}_render.txt", this_render_depth)
#                        np.savetxt(f"./simpts/{str(batch_idx)}_{str(b_idx).zfill(2)}_kps_reconstruct.txt", reconstruct_pt)
                
#                if dr_loss == "mse":
#                    if dr_input == "sim":
#                        res = DPRenderer.loss_forward(batch_gt_simdepth, batch_gt_mask,img_path=batch["rgb_path"], update_idx=update_idx, link_idx=link_idx, dr_order=dr_order)
#                    elif dr_input == "pf":
#                        res = DPRenderer.loss_forward(batch_gt_pfdepth, batch_gt_mask,img_path=batch["rgb_path"], update_idx=update_idx, link_idx=link_idx, dr_order=dr_order)
#                    else:
#                        raise ValueError
#                elif dr_loss == "chamfer":
#                    if dr_input == "sim":
#                        res = DPRenderer.loss_forward_chamfer(batch_gt_simxyz, batch_gt_mask,img_path=batch["rgb_path"], update_idx=update_idx, link_idx=link_idx, dr_order=dr_order)
#                    elif dr_input == "pf":
#                        res = DPRenderer.loss_forward_chamfer(batch_gt_pfxyz, batch_gt_mask,img_path=batch["rgb_path"], update_idx=update_idx, link_idx=link_idx, dr_order=dr_order)
#                    else:
#                        raise ValueError
#                else:
#                    raise ValueError
#                
#                all_res[:, :, :, update_idx *dr_w : (update_idx+1) * dr_w] = res.transpose(0, 3, 1, 2)
#                DPRenderer.loss_backward(optimizer_list)


#                batch_acc_from_joint_angles(joints_angle_pred[:, link_idx-1:link_idx, :].detach().cpu().numpy(), joints_1d_gt[:, link_idx-1:link_idx, None].detach().cpu().numpy(), acc_thresholds)
                
            
#            grid_image = make_grid(torch.from_numpy(all_res), 1, normalize=False, scale_each=False) 
#            if batch_idx <= 30:
#                cv2.imwrite(os.path.join(blended_path, f"blend_cuda{device}_{str(batch_idx).zfill(4)}.png"), grid_image.detach().cpu().numpy().transpose(1,2,0)[:,:,::-1])
#            
##            batch_dt_quaternion = batch_dt_quaternion / torch.norm(batch_dt_quaternion, dim=-1,keepdim=True)
##            batch_final_rot = quaternion_to_matrix(batch_dt_quaternion)
##            pose_pred[:, :3, :3] = batch_final_rot
#            
#            joints_3d_rob_pred = compute_3n_loss_42(joints_angle_pred, input_K.device)
#            pose_pred_clone = []
#            for b_idx in range(b):
#                pose_pred_clone.append(compute_rede_rt(joints_3d_rob_pred[b_idx:b_idx+1, :, :], joints_3d_pred_norm[b_idx:b_idx+1, :, :]))
#            pose_pred_clone = torch.cat(pose_pred_clone)
#            pose_pred_clone[:, :3, 3] = joints_3d_pred_tensor[:, 0] + pose_pred_clone[:, :3, 3] 
#            #pose_pred_clone[:, :3, 3] = batch_dt_trans
#            pose_pred = pose_pred_clone

            
        joints_angle_pred_lst.append(joints_angle_pred)
        pose_pred_lst.append(pose_pred)
        joints_3d_pred_lst.append(torch.from_numpy(joints_3d_pred).to(device))
        joints_angle_gt_lst.append(joints_1d_gt)
        joints_3d_gt_lst.append(joints_3d_gt)
        pose_gt_lst.append(pose_gt) 
        
        
        #pose_pred = batch_outlier_removal_pose(joints_2d_depth, joints_3d_rob_pred, joints_3d_pred_norm, joints_3d_pred_tensor, (h, w))
        joints_3d_err_lst.append(compute_3d_error(joints_3d_gt-joints_3d_gt[:, :1, :], joints_3d_pred_norm))
        
        # 
        joints_kps_3d_pred = compute_kps_joints_loss(pose_pred[:, :3, :3], (pose_pred[:, :3, 3])[:, :, None], joints_angle_pred, device)  
        #joints_kps_3d_pred = compute_kps_joints_loss(pose_pred[:, :3, :3], (pose_pred[:, :3, 3])[:, :, None], joints_1d_gt[:, :, None], device)   
        #joints_kps_3d_pred = compute_kps_joints_loss(batch["R2C_Pose"][:, :3, :3].float().to(device), batch["R2C_Pose"][:, :3, 3:4].float().to(device), joints_1d_gt[:, :, None], device)
        
        
        kps_add = kps_add + batch_add_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy())
        kps_mAP.append(batch_mAP_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy(),thresholds))

        
        kps_pred_lst.append(joints_kps_3d_pred)
        kps_gt_lst.append(joints_kps_3d_gt)
        
        # joints_pred, joints_gt = joints_3d_pred, joints_3d_gt.detach().cpu().numpy() # B x N x 3
        joints_pred, joints_gt = joints_3d_pred_tensor.detach().cpu().numpy(), joints_3d_gt.detach().cpu().numpy() # B x N x 3
        
        
        ass_add_mean = batch_add_from_pose(joints_pred, joints_gt)
        ass_add = ass_add + ass_add_mean
        
        ass_mAP_mean = batch_mAP_from_pose(joints_pred, joints_gt,thresholds) # 
        ass_mAP.append(ass_mAP_mean)
        
        if link_idx == "whole" or link_idx == 0:
            angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred.detach().cpu().numpy(), joints_1d_gt.detach().cpu().numpy()[:, :, None], acc_thresholds)
        elif link_idx >= 1 and dr_order == "single":
            angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred[:, link_idx-1:link_idx,:].detach().cpu().numpy(), joints_1d_gt[:, link_idx-1:link_idx, None].detach().cpu().numpy(), acc_thresholds)
        elif link_idx >= 1 and dr_order == "sequence":
            angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred[:, :link_idx,:].detach().cpu().numpy(), joints_1d_gt[:, :link_idx, None].detach().cpu().numpy(), acc_thresholds)
        else:
            angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred.detach().cpu().numpy(), joints_1d_gt.detach().cpu().numpy()[:, :, None], acc_thresholds)
        angles_acc.append(angles_acc_mean)
    
    
    torch.distributed.barrier()
    ass_add = distributed_concat(torch.from_numpy(np.array(ass_add)).to(device), len(test_sampler.dataset))
    ass_mAP = distributed_concat(torch.from_numpy(np.array(ass_mAP)).to(device), len(test_sampler.dataset))
    angles_acc = distributed_concat(torch.from_numpy(np.array(angles_acc)).to(device), len(test_sampler.dataset))
    kps_add  = distributed_concat(torch.from_numpy(np.array(kps_add)).to(device), len(test_sampler.dataset))
    kps_mAP = distributed_concat(torch.from_numpy(np.array(kps_mAP)).to(device), len(test_sampler.dataset))
    uv_pck  = distributed_concat(torch.from_numpy(np.array(uv_pck)).to(device), len(test_sampler.dataset))
    z_pck = distributed_concat(torch.from_numpy(np.array(z_pck)).to(device), len(test_sampler.dataset))
    pred_z = distributed_concat(torch.from_numpy(np.array(pred_z)).to(device), len(test_sampler.dataset))
    gt_z = distributed_concat(torch.from_numpy(np.array(gt_z)).to(device), len(test_sampler.dataset))

    
    joints_3d_pred_gather = distributed_concat(torch.cat(joints_3d_pred_lst, dim=0), len(test_sampler.dataset))
    joints_angle_pred_gather = distributed_concat(torch.cat(joints_angle_pred_lst, dim=0), len(test_sampler.dataset))
    pose_pred_gather = distributed_concat(torch.cat(pose_pred_lst, dim=0), len(test_sampler.dataset))
    joints_3d_gt_gather = distributed_concat(torch.cat(joints_3d_gt_lst, dim=0), len(test_sampler.dataset))
    joints_angle_gt_gather = distributed_concat(torch.cat(joints_angle_gt_lst, dim=0), len(test_sampler.dataset))
    pose_gt_gather = distributed_concat(torch.cat(pose_gt_lst, dim=0), len(test_sampler.dataset))
    
    kps_pred_gather = distributed_concat(torch.cat(kps_pred_lst, dim=0), len(test_sampler.dataset))
    kps_gt_gather = distributed_concat(torch.cat(kps_gt_lst, dim=0), len(test_sampler.dataset))
    
    joints_3d_err_gather = distributed_concat(torch.cat(joints_3d_err_lst, dim=1), len(test_sampler.dataset), dim=1)

    
#    if dist.get_rank() == 0:
#        print("joints_3d_pred_gather", joints_3d_pred_gather.shape) # B x num_kps x 3
#        print("pose_pred_gather", pose_pred_gather.shape) # B x 3 x 4
#        print("joints_angle_pred_gather", joints_angle_pred_gather.shape) # B x num_joints
#        print("joints_3d_gt_gather", joints_3d_gt_gather.shape) # B x num_kps x 3
#        print("pose_gt_gather", pose_gt_gather.shape) # B x 3 x 4
#        print("joints_angle_gt_gather", joints_angle_gt_gather.shape) # B x num_joints
#        print("ass_add", ass_add.shape)
#        print("ass_mAP", ass_mAP.shape)
#        print("angles_acc", angles_acc.shape)
#        
    
    
    ass_add = ass_add.detach().cpu().numpy().tolist()
    ass_mAP = ass_mAP.detach().cpu().numpy().tolist()
    angles_acc = angles_acc.detach().cpu().numpy().tolist()
    kps_add = kps_add.detach().cpu().numpy().tolist()
    kps_mAP = kps_mAP.detach().cpu().numpy().tolist()
    uv_pck = uv_pck.detach().cpu().numpy().tolist()
    z_pck = z_pck.detach().cpu().numpy().tolist()
    pred_z = pred_z.detach().cpu().numpy().tolist()
    gt_z = gt_z.detach().cpu().numpy().tolist()
    
    angles_results = np.round(np.mean(angles_acc,axis=0)*100, 2)
    angles_dict = dict()
    kps_add_results = add_metrics(kps_add, add_thresh)
    kps_mAP_results = np.round(np.mean(kps_mAP, axis=0) * 100, 2)
    kps_mAP_dict = dict()
    ass_add_results = add_metrics(ass_add, add_thresh)
    ass_mAP_results = np.round(np.mean(ass_mAP, axis=0) * 100, 2)
    ass_mAP_dict = dict()
    
    uv_pck_results = pck_metrics(uv_pck)
    z_pck_results = pck_metrics(z_pck)
    
#    ass_add_results = kps_add_results
#    ass_mAP_results = kps_mAP_results
#    ass_mAP_dict = dict()
    
    
    for b_ in range(joints_3d_err_gather.shape[0]):
        np.savetxt(os.path.join(info_path, f"{str(b_).zfill(2)}_err.txt"), joints_3d_err_gather[b_].detach().cpu().numpy() * 1000)
    
    exp_id = cfg["EXP_ID"]
    epoch_id = cfg["EPOCH_ID"]
    cx_delta = cfg["CX_DELTA"]
    cy_delta = cfg["CY_DELTA"] 
    
    file_name = os.path.join(results_path, f"iter_{str(dr_iter_num)}_{testing_data_dir.split('/')[-2]}_{dr_loss}_{dr_input}_{dr_order}_cx{str(cx_delta)}_cy{str(cy_delta)}.txt")
#    pred_z_path = os.path.join(results_path, f"iter_{str(dr_iter_num)}_{testing_data_dir.split('/')[-2]}_{dr_loss}_{dr_input}_{dr_order}_cx{str(cx_delta)}_cy{str(cy_delta)}_predz.txt")
#    np.savetxt(pred_z_path, np.array(pred_z))
#    np.savetxt(os.path.join(results_path, f"iter_{str(dr_iter_num)}_{testing_data_dir.split('/')[-2]}_{dr_loss}_{dr_input}_{dr_order}_cx{str(cx_delta)}_cy{str(cy_delta)}_gtz.txt"), np.array(gt_z))
    prediction_gt_name = os.path.join(info_path, f"iter_{str(dr_iter_num)}_link_{link_idx}_{testing_data_dir.split('/')[-2]}_{dr_loss}_{dr_input}.json") 
    
    if dist.get_rank() == 0:
        write_prediction_and_gt(prediction_gt_name, joints_3d_pred_gather, joints_3d_gt_gather, 
                                         pose_pred_gather, pose_gt_gather, 
                                         joints_angle_pred_gather, joints_angle_gt_gather,
                                         kps_pred_gather, kps_gt_gather
                                          )
        
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
            
            # print PCK
            print_to_screen_and_file(
                f, " UV PCK AUC: {:.5f}".format(uv_pck_results["l2_error_auc"])
            )
            print_to_screen_and_file(
                f,
                   " UV PCK  AUC threshold: {:.5f} m".format(uv_pck_results["l2_error_auc_thresh_px"]),
            )
            print_to_screen_and_file(
                f, " UV PCK  Mean: {:.5f}".format(uv_pck_results["l2_error_mean_px"])
            )
            print_to_screen_and_file(
                f, " UV PCK  Median: {:.5f}".format(uv_pck_results["l2_error_median_px"])
            )
            print_to_screen_and_file(
                f, " UV PCK  Std Dev: {:.5f}".format(uv_pck_results["l2_error_std_px"]))
            print_to_screen_and_file(f, "")
            
            print_to_screen_and_file(
                f, " Z PCK AUC: {:.5f}".format(z_pck_results["l2_error_auc"])
            )
            print_to_screen_and_file(
                f,
                   " Z PCK  AUC threshold: {:.5f} m".format(z_pck_results["l2_error_auc_thresh_px"]),
            )
            print_to_screen_and_file(
                f, " Z PCK  Mean: {:.5f}".format(z_pck_results["l2_error_mean_px"])
            )
            print_to_screen_and_file(
                f, " Z PCK  Median: {:.5f}".format(z_pck_results["l2_error_median_px"])
            )
            print_to_screen_and_file(
                f, " Z PCK  Std Dev: {:.5f}".format(z_pck_results["l2_error_std_px"]))
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
            
            # print pck
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
            
            
        
    
            
    return ass_add_results, ass_mAP_dict, angles_dict
            
            
if __name__ == "__main__":
    cfg, args = update_config()
    torch.multiprocessing.set_start_method('spawn')
    if cfg["LOCAL_RANK"] != -1:
        torch.cuda.set_device(cfg["LOCAL_RANK"])
        device=torch.device("cuda",cfg["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_whole_spdh_model(cfg, device)
    model = model.to(device)
    num_gpus = torch.cuda.device_count()
    epoch_id = cfg["EPOCH_ID"]
    exp_id = cfg["EXP_ID"]
    print("device", device)
    #if num_gpus > 1:
    print('use {} gpus!'.format(num_gpus))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    
    
    start_epoch, global_iter = 0, 0
    optimizer, scheduler = init_optimizer(model, cfg)
    
    
    if cfg["MODEL_PATH"]:
        path = cfg["MODEL_PATH"]
        model, optimizer, scheduler, start_epoch, global_iter = load_spdh_model(model, optimizer, scheduler, path, device)
        print("path", path)
    
    if cfg["SYN_TEST"]:
        cfg["DATASET"]["TESTING_ROOT"] = cfg["SYN_TEST"]
        
    #model = model.to(device)
    network_inference(model, cfg, epoch_id, device)
        
    
    
    
















