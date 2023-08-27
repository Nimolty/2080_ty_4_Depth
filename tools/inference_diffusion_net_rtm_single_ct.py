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
from PIL import Image as PILImage
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
from depth_c2rp.utils.spdh_utils import reduce_mean, init_spdh_model, compute_kps_joints_loss, solve_pnp_ransac, solve_pnp, overlay_points_on_image
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred, log_and_visualize_single
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat
from depth_c2rp.utils.analysis import flat_add_from_pose, add_metrics, add_from_pose, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles, batch_outlier_removal_pose, batch_repeat_add_from_pose, batch_repeat_mAP_from_pose, batch_repeat_acc_from_joint_angles, batch_pck_from_pose, pck_metrics
from depth_c2rp.utils.rgb_utils import batch_affine_transform_pts
#from depth_c2rp.datasets.datasets_diffusion_inference_rtm_ctcopy import Diff_dataset
from depth_c2rp.datasets.datasets_diffusion_inference_rtm_ct import Diff_dataset
from depth_c2rp.models.mmpose_ours.model import build_mmpose_network

# diffusion
from depth_c2rp.diffusion_utils.diffusion_network import build_diffusion_network, build_simple_network, load_simplenet_model, load_single_simplenet_model
from depth_c2rp.diffusion_utils.diffusion_losses import get_optimizer, optimization_manager
from depth_c2rp.diffusion_utils.diffusion_ema import ExponentialMovingAverage
from depth_c2rp.diffusion_utils.diffusion_utils import diff_save_weights, diff_load_weights, draw_heatmap_simcc
from depth_c2rp.diffusion_utils.diffusion_sampling import get_sampling_fn

def load_single_model(model, path, device):
    state_dict = {}
    checkpoint = torch.load(path, map_location=device)["model"]
    for key, value in checkpoint.items():
        #print(key)
        if key[:7] == "module.":
            new_key = key[7:]
        else:
            raise NotImplementedError
        state_dict[new_key] = value
    model.load_state_dict(state_dict, strict=True)
    print(f'restored "{path}" model. Key errors:')
    return model



def main(cfg, mAP_thresh=[0.02, 0.11, 0.01], add_thresh=0.06,angles_thresh=[2.5, 30.0, 2.5]):
    set_random_seed(int(cfg["DIFF_MODEL"]["SEED"]))
    assert type(cfg) == dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build DataLoader
    sigma_pixel = cfg["SIGMA_PIXEL"]
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
                                     joint_names=[f"panda_joint_3n_{i+1}" for i in range(14)],
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
                                     cond_norm=train_cfg["COND_NORM"],
                                     mean=model_cfg["MEAN"],
                                     std=model_cfg["STD"],
                                     )    

    training_dataset.train()
    training_loader = DataLoader(training_dataset, shuffle=False, batch_size=eval_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True) 

    val_dataset = copy.copy(training_dataset)  
    val_dataset.eval()    
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1,
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    real_dataset = copy.copy(training_dataset)
    real_dataset.real()
    real_loader = DataLoader(real_dataset, shuffle=False, batch_size=1,
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
                                  
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    checkpoint_path = os.path.join(save_path, "CHECKPOINT")
    tb_path = os.path.join(save_path, "TB")
    heatmap_id = cfg["RESUME_HEATMAP"].split('/')[-2][8:11]
    pred_mask_flag = cfg["PRED_MASK"]
    results_path = os.path.join(save_path, "ct", f"NUMERICAL_RESULTS_{heatmap_id}_PRED_MASK_{pred_mask_flag}")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    exists_or_mkdir(results_path)


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

    # Build Model 
    model = build_diffusion_network(cfg, device)
    model = model.to(device)
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
    simplenet_model = simplenet_model.to(device)

    if "RGB" in cfg["DIFF_MODEL"]["INPUT_TYPE"]:
        inchannels = 7
    
    rtm_model = build_mmpose_network(name=cfg["RTM"]["MODEL_NAME"],
                                     num_keypoints=cfg["DIFF_MODEL"]["NUM_JOINTS"], 
                                     input_size=cfg["DATASET"]["OUTPUT_RESOLUTION"],
                                     in_channels=inchannels
                                     )        
    rtm_model = rtm_model.to(device) 

    if cfg["RESUME_SIMPLENET"] != "":
        simplenet_model = load_single_simplenet_model(simplenet_model, cfg["RESUME_SIMPLENET"], device)
    if cfg["RESUME_HEATMAP"] != "":
        rtm_model = load_single_model(rtm_model,cfg["RESUME_HEATMAP"],device)
        print("successfully loading rtm_model!") 

    # Build Opimizer
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg["DIFF_MODEL"]["EMA_RATE"])
    optimizer = get_optimizer(cfg, model.parameters()) 

    if cfg["RESUME"]:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        ema.load_state_dict(checkpoint['ema'])
        global_iter = checkpoint['step'] 

    # Build Loss Function
    loss_cfg = cfg["LOSS"]
    num_samples = cfg["DIFF_EVAL"]["NUM_SAMPLES"]
    method = cfg["DIFF_SAMPLING"]["METHOD"]

    print("len_dataloader", len(training_loader))
    print("num_samples", num_samples)
    
    print("len_dataloader", len(training_loader))
    print("num_samples", num_samples)
#    with torch.no_grad():
    gt_angle_flag = train_cfg["GT_ANGLE_FLAG"]
    pred_2d_flag = cfg["PRED_2D_FLAG"]
    cond_norm_flag = train_cfg["COND_NORM"]
    real_name = dataset_cfg["REAL_ROOT"].split('/')[-2]
    #split = {"Real" : [real_sampler, real_loader, real_dataset_dir], "Validation" : [val_sampler, val_loader, val_dataset_dir]}
    #split = {"Validation" : [val_sampler, val_loader, val_dataset_dir]}
    split = {"Real" : [real_loader, real_dataset_dir],}
    rtm_model.eval()
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
    ass_add_per = []
    angles_acc_per = []
    info_path = os.path.join(save_path, "ct", "INFERENCE_LOGS", str(start_epoch).zfill(5))
    exists_or_mkdir(info_path)  

    with torch.no_grad():
        for mode, value in split.items():
            loader = value[0]
            exists_or_mkdir(f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/DIFF/diff_37/vis/{real_name}/")
            for batch_idx, batch in enumerate(tqdm(loader)):
                # if batch_idx > 30:
                #     break

                joints_3d_gt = batch["next_joints_3D_Z"].to(device, non_blocking=True).float()
                joints_kps_3d_gt = batch["next_joints_3D_kps"].to(device, non_blocking=True).float()
                joints_1d_gt = batch["next_joints_7"].to(device).float()
                pose_gt = batch["next_R2C_Pose"].to(device).float()
                intrinsic = batch["intrinsic"].to(device).float()
                fx, fy = intrinsic[0, 0, 0], intrinsic[0, 1, 1]
                cx, cy = intrinsic[0, 0, 2], intrinsic[0, 1, 2]
                bbox = batch["bbox"].detach().cpu().numpy() # B x 4
                trans_output_inverse = batch["trans_output_inverse"].detach().cpu().numpy()    
                trans_output = batch["trans_output"].detach().cpu().numpy()
                rgb_file = batch["rgb_path"][0] 
                rgb_image = PILImage.open(rgb_file).convert("RGB")

                t1 = time.time()
                if pred_2d_flag:
                    prev_rgb_img_input = batch["prev_rgb_img_input"].to(device).float()
                    next_rgb_img_input = batch["next_rgb_img_input"].to(device).float()
                    uv_gt = batch["next_joints_2D"].to(device).float()
                    if batch_idx == 0:
                        prev_heatmaps = torch.zeros(1, 1, next_rgb_img_input.shape[2], next_rgb_img_input.shape[3]).to(device).float()
                    else:
                        uv_pred_prev = uv_pred.detach().cpu().numpy().copy()
                        uv_pred_prev[:, :, 0] -= np.repeat(bbox[:, 1:2], n_kp, axis=1)
                        uv_pred_prev[:, :, 1] -= np.repeat(bbox[:, 0:1], n_kp, axis=1)    
                        uv_pred_prev = batch_affine_transform_pts(uv_pred_prev, trans_output)
                        heatmaps_pred = draw_heatmap_simcc(uv_pred_prev, output_img_size=dataset_cfg["OUTPUT_RESOLUTION"],device=device, sigma_pixel=sigma_pixel)
                        prev_heatmaps = torch.sum(heatmaps_pred, dim=1, keepdims=True)

                    input_tensor = torch.cat([prev_rgb_img_input, next_rgb_img_input, prev_heatmaps], dim = 1)
                    
                    joints_2d_pred = rtm_model.predict(input_tensor)
                    joints_2d_pred = torch.from_numpy(joints_2d_pred).float().to(device)
                    n_kp = joints_2d_pred.shape[1]
                    uv_pred = batch_affine_transform_pts(joints_2d_pred.detach().cpu().numpy(), trans_output_inverse) # B x N x 2
                    uv_pred[:, :, 0] += np.repeat(bbox[:, 1:2], n_kp, axis=1)
                    uv_pred[:, :, 1] += np.repeat(bbox[:, 0:1], n_kp, axis=1)
                    uv_pred = torch.from_numpy(uv_pred).float().to(device)

                    uv_pred_list.append(uv_pred)
                    uv_gt_list.append(uv_gt)
                    uv_pck = batch_pck_from_pose(uv_gt.detach().cpu().numpy(), uv_pred.detach().cpu().numpy())
                    uv_kps_pck = torch.linalg.norm((uv_gt-uv_pred), dim=-1) # B x N
                    uv_kps_pck_list.append(uv_kps_pck.detach().cpu().numpy().tolist())
                    uv_pck_list.append(uv_pck)    

                    if cond_norm_flag:
                        batch["next_joints_2D_cond_norm"][:, :, 0] = (uv_pred[:, :, 0] - cx) / fx
                        batch["next_joints_2D_cond_norm"][:, :, 1] = (uv_pred[:, :, 1] - cy) / fy
                    else:
                        raise NotImplementedError  

                joints_2d = batch["next_joints_2D_cond_norm"].float().to(device, non_blocking=True)
                bs, N, _ = joints_2d.shape
                joints_2d_yummy = torch.zeros(bs, N, 1).to(device, non_blocking=True)
                joints_2d = torch.cat([joints_2d, joints_2d_yummy], dim=-1).float()    

                joints_3d_gt_repeat = joints_3d_gt.repeat(num_samples, 1, 1)
                joints_2d_repeat = joints_2d.repeat(num_samples, 1, 1)
                joints_1d_gt_repeat = joints_1d_gt.repeat(num_samples, 1)
                pose_gt_repeat = pose_gt.repeat(num_samples, 1, 1)
                joints_kps_3d_gt_repeat = joints_kps_3d_gt.repeat(num_samples, 1, 1) 
                
                # image_points = uv_pred[0].detach().cpu().numpy().tolist() + uv_gt[0].detach().cpu().numpy().tolist()
                # annotation_color_dot = ["red"] * len(uv_pred[0]) + ["green"] * (len(uv_gt[0]))
                # whole_rgb_image = overlay_points_on_image(rgb_image, image_points=image_points, annotation_color_dot = annotation_color_dot, point_diameter=4)
                # whole_rgb_image.save(f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/DIFF/diff_37/vis/{real_name}/{str(batch_idx).zfill(4)}.png")
                
                # pred_rgb_image = overlay_points_on_image(rgb_image, image_points=uv_pred[0].detach().cpu().numpy().tolist(), annotation_color_dot = ["red"] * len(uv_pred[0]), point_diameter=4)
                # pred_rgb_image.save(f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/DIFF/diff_37/vis/{real_name}/pred_{str(batch_idx).zfill(4)}.png")
                ema.copy_to(model.parameters())
                t3 = time.time()
                # if batch_idx == 0:
                #     trajs, results = model.sampling_fn(
                #                 model.model,
                #                 condition=joints_2d_repeat,
                #                 num_samples=num_samples,
                #                 track_idx = batch_idx
                #             )  # [b ,j ,3]    
                # else: 
                #     trajs, results = model.sampling_fn(
                #                 model.model,
                #                 condition=joints_2d_repeat,
                #                 num_samples=num_samples,
                #                 z=joints_3d_pred_prev,
                #                 track_idx = batch_idx
                #             ) 
 
                trajs, results = model.sampling_fn(
                                model.model,
                                condition=joints_2d_repeat,
                                num_samples=num_samples,
                                track_idx = 0
                            ) 
                
                joints_3d_pred_repeat = results[-1].to(device)
                joints_3d_pred_prev = joints_3d_pred_repeat.clone()
                joints_3d_pred_repeat = joints_3d_pred_repeat.reshape(num_samples, bs, -1, 3) 
                joints_3d_pred_repeat = joints_3d_pred_repeat.permute(1, 0, 2, 3) # bs x num_samples x N x 3
                joints_3d_pred = torch.mean(joints_3d_pred_repeat, dim=1)
                joints_3d_pred_repeat = joints_3d_pred_repeat.permute(0, 2, 1, 3) # bs x N x num_samples x 3
                joints_angle_pred, pose_pred = simplenet_model(joints_3d_pred.clone(), joints_1d_gt[..., None], gt_angle_flag) 
                
                t2 = time.time()

                try:
                    results_repeat = torch.cat(trajs,dim=0).reshape(-1, num_samples, bs, N, 3) # steps x num_samples x bs x N x 3
                    results_repeat = results_repeat.permute(2, 1, 0, 3, 4) # bs x num_samples x steps x N x 3
                except: 
                    results_repeat = trajs                                   
    
                if gt_angle_flag:
                    joints_angle_pred = joints_1d_gt[..., None]  
                
                joints_angle_pred_lst.append(joints_angle_pred)
                joints_angle_gt_lst.append(joints_1d_gt)
                joints_3d_pred_lst.append(joints_3d_pred)
                joints_3d_gt_lst.append(joints_3d_gt)
                pose_pred_lst.append(pose_pred)
                pose_gt_lst.append(pose_gt) 
                joints_3d_pred_repeat_lst.append(joints_3d_pred_repeat)
                
                
                joints_kps_3d_pred = compute_kps_joints_loss(pose_pred[:, :3, :3], (pose_pred[:, :3, 3])[:, :, None], joints_angle_pred, device)  
                kps_add = kps_add + batch_add_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy())
                #print("kps_add", kps_add)
                kps_mAP.append(batch_mAP_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy(), thresholds))
                kps_pred_lst.append(joints_kps_3d_pred)
                kps_gt_lst.append(joints_kps_3d_gt)
                ass_add_mean = batch_add_from_pose(joints_3d_pred.detach().cpu().numpy(), joints_3d_gt.detach().cpu().numpy())
                ass_add = ass_add + ass_add_mean  
                this_ass_add_per = torch.linalg.norm((joints_3d_pred - joints_3d_gt), dim=-1)
                ass_add_per.append(this_ass_add_per.detach().cpu().numpy().tolist())
                ass_mAP_mean = batch_mAP_from_pose(joints_3d_pred.detach().cpu().numpy(), joints_3d_gt.detach().cpu().numpy(), thresholds) # 
                ass_mAP.append(ass_mAP_mean)
                angles_acc_mean = batch_acc_from_joint_angles(joints_angle_pred.detach().cpu().numpy(), joints_1d_gt.detach().cpu().numpy()[:, :, None], acc_thresholds)
                this_angles_acc_per = torch.linalg.norm((joints_angle_pred - joints_1d_gt[..., None]), dim=-1)
                angles_acc_per.append(this_angles_acc_per.detach().cpu().numpy().tolist())
                angles_acc.append(angles_acc_mean) 

            angles_acc = np.concatenate(angles_acc, axis=0)
            ass_mAP = np.concatenate(ass_mAP, axis=0)
            kps_mAP = np.concatenate(kps_mAP, axis=0)
            
            ass_add_per = np.concatenate(ass_add_per)
            angles_acc_per = np.concatenate(angles_acc_per)

            if pred_2d_flag:
                uv_pck_list = np.concatenate(uv_pck_list, axis=0) 
                uv_kps_pck_list = np.concatenate(uv_kps_pck_list, axis=0)            
            
            angles_results = np.round(np.mean(angles_acc,axis=0)*100, 2)
            angles_dict = dict()
            kps_add_results = add_metrics(kps_add, add_thresh)
            kps_mAP_results = np.round(np.mean(kps_mAP, axis=0) * 100, 2)
            kps_mAP_dict = dict()
            ass_add_results = add_metrics(ass_add, add_thresh)
            ass_mAP_results = np.round(np.mean(ass_mAP, axis=0) * 100, 2)
            ass_mAP_dict = dict()
            pck_results = pck_metrics(uv_pck_list)

            change_intrinsic_flag = cfg["CHANGE_INTRINSIC"]
            #real_name = dataset_cfg["VAL_ROOT"].split('/')[-2]
            file_name = os.path.join(results_path, f"Epoch_{str(start_epoch).zfill(5)}_{real_name}_change_intrin_{change_intrinsic_flag}_angle_{gt_angle_flag}_ns_{num_samples}_pred2d_{pred_2d_flag}_{method}.txt")
            path_meta = os.path.join(info_path, f"Epoch_{str(start_epoch).zfill(5)}_{real_name}_change_intrin_{change_intrinsic_flag}_angle_{gt_angle_flag}_ns_{num_samples}_pred2d_{pred_2d_flag}_{method}.json")                            

            with open(file_name, "w") as f:
                print_to_screen_and_file(
                f, "Analysis results for dataset: {}".format(split[mode][1])
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
                
                print_to_screen_and_file(
                    f, f" ass_add_per : {np.mean(ass_add_per, axis=0)}")
                    
                print_to_screen_and_file(
                    f, f" angles_acc_per : {np.mean(angles_acc_per, axis=0)}")
                
                
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
    cfg, args = update_config()
    main(cfg)













































