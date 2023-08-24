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
from depth_c2rp.utils.spdh_utils import reduce_mean, init_spdh_model, compute_kps_joints_loss, solve_pnp_ransac, solve_pnp
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred, log_and_visualize_single
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat
from depth_c2rp.utils.analysis import flat_add_from_pose, add_metrics, add_from_pose, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles, batch_outlier_removal_pose, batch_repeat_add_from_pose, batch_repeat_mAP_from_pose, batch_repeat_acc_from_joint_angles, batch_pck_from_pose, pck_metrics
from depth_c2rp.utils.rgb_utils import batch_affine_transform_pts
from depth_c2rp.datasets.datasets_diffusion_inference_rtm import Diff_dataset
from depth_c2rp.models.mmpose_ours.model import build_mmpose_network


def load_single_model(model, path, device):
    state_dict = {}
    checkpoint = torch.load(path, map_location=device)["model"]
    for key, value in checkpoint.items():
        if key[:7] == "module.":
            new_key = key[7:]
        else:
            raise ValueError
        state_dict[new_key] = value
    model.load_state_dict(state_dict, strict=True)
    print(f'restored "{path}" model. Key errors:')
    return model



def main(cfg, mAP_thresh=[0.02, 0.11, 0.01], add_thresh=0.06,angles_thresh=[2.5, 30.0, 2.5]):
    """
        This code is designed for the whole training. 
    """
    # expect cfg is a dict
    set_random_seed(int(cfg["DIFF_MODEL"]["SEED"]))
    
    assert type(cfg) == dict
    device_ids = [4,5,6,7] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
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
    #train_sampler = DistributedSampler(training_dataset)
    training_loader = DataLoader(training_dataset, shuffle=False, batch_size=eval_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True) 
    
    
    val_dataset = copy.copy(training_dataset)  
    val_dataset.eval()    
    #val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    real_dataset = copy.copy(training_dataset)
    real_dataset.real()
    #real_sampler = DistributedSampler(real_dataset)
    real_loader = DataLoader(real_dataset, shuffle=False, batch_size=eval_cfg["BATCH_SIZE"],
                                  num_workers=4, pin_memory=True, drop_last=False)
    

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


    start_epoch,global_iter = 0, 0

    
    num_gpus = torch.cuda.device_count()
    print("device", device)
    #if num_gpus > 1:
    print('use {} gpus!'.format(num_gpus))

    if cfg["DIFF_MODEL"]["INPUT_TYPE"] == "D":
        inchannels = 1
    if "XYZ" in cfg["DIFF_MODEL"]["INPUT_TYPE"]:
        inchannels = 3
    if "RGB" in cfg["DIFF_MODEL"]["INPUT_TYPE"]:
        inchannels = 3
    if "CONCAT" in cfg["DIFF_MODEL"]["INPUT_TYPE"]:
        inchannels += 1
    rtm_model = build_mmpose_network(name=cfg["RTM"]["MODEL_NAME"],
                                     num_keypoints=cfg["DIFF_MODEL"]["NUM_JOINTS"], 
                                     input_size=cfg["DATASET"]["OUTPUT_RESOLUTION"],
                                     in_channels=inchannels
                                     )

    rtm_model = rtm_model.to(device)

    if cfg["RESUME_HEATMAP"] != "":
        rtm_model = load_single_model(rtm_model, cfg["RESUME_HEATMAP"], device)
        print("successfully loading rtm_model!")    

    # Build Loss Function
    loss_cfg = cfg["LOSS"]
    num_samples = cfg["DIFF_EVAL"]["NUM_SAMPLES"]
    method = cfg["DIFF_SAMPLING"]["METHOD"]

    print("len_dataloader", len(training_loader))
    print("num_samples", num_samples)
    gt_angle_flag = train_cfg["GT_ANGLE_FLAG"]
    pred_2d_flag = cfg["PRED_2D_FLAG"]
    cond_norm_flag = train_cfg["COND_NORM"]
    split = {"Real" : [real_loader, real_dataset_dir],}
    rtm_model.eval()
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
    info_path = os.path.join(save_path, "INFERENCE_LOGS", str(start_epoch).zfill(5))
    exists_or_mkdir(info_path)

    with torch.no_grad():
        for mode, value in split.items():
            loader = value[0]
            
            for batch_idx, batch in enumerate(tqdm(loader)):
                batch_json = {}
                curr_loss = 0.0
                
                
                #input_tensor = batch["xyz_img_scale"].float().to(device)
                #xyz_img = batch["xyz_img"].detach().cpu().numpy()
                joints_3d_gt = batch['joints_3D_Z'].to(device, non_blocking=True).float()
                joints_kps_3d_gt = batch["joints_3D_kps"].to(device, non_blocking=True).float()
                joints_1d_gt = batch["joints_7"].to(device).float()
                pose_gt = batch["R2C_Pose"].to(device).float()
                intrinsic = batch["intrinsic"].to(device).float()
                intrinsic_np = batch["intrinsic"].detach().cpu().numpy()
                joints_14_rob_gt = batch["joints_3D_Z_rob"].detach().cpu().numpy()
                fx, fy = intrinsic[0, 0, 0], intrinsic[0, 1, 1]
                cx, cy = intrinsic[0, 0, 2], intrinsic[0, 1, 2]
                bbox = batch["bbox"].detach().cpu().numpy() # B x 4
                trans_output_inverse = batch["trans_output_inverse"].detach().cpu().numpy()
                input_uv_off = batch["joints_2D_output_off"].to(device).float()
                joints_2D_output_ind = batch["joints_2D_output_ind"].to(device).type(torch.int64)
                depth_path = batch["meta_path"]
                
                depth_path_lst += depth_path


                torch.cuda.synchronize()
                t1 = time.time()
                if pred_2d_flag:
                    input_tensor = batch["input_tensor"].float().to(device)
                    uv_gt = batch['joints_2D'].to(device).float()
                    gt_x = batch["keypoint_x_labels"].float().to(device)
                    gt_y = batch["keypoint_y_labels"].float().to(device)
                    keypoint_weights = batch["keypoint_weights"].float().to(device)
                    
                    joints_2d_pred = rtm_model.predict(input_tensor)
                    joints_2d_pred = torch.from_numpy(joints_2d_pred).float().to(device)
                    n_kp = joints_2d_pred.shape[1]
                    uv_pred_np = batch_affine_transform_pts(joints_2d_pred.detach().cpu().numpy(), trans_output_inverse) # B x N x 2
                    uv_pred_np[:, :, 0] += np.repeat(bbox[:, 1:2], n_kp, axis=1)
                    uv_pred_np[:, :, 1] += np.repeat(bbox[:, 0:1], n_kp, axis=1)
                    uv_pred = torch.from_numpy(uv_pred_np).float().to(device)
                                        
                    bs = uv_pred.shape[0]
                    pose_pred = np.zeros((bs, 4, 4))
                    for b in range(bs):
                        #pnp_retval, translation, quaternion = solve_pnp(joints_14_rob_gt[b], uv_pred_np[b], intrinsic_np[b])
                        pnp_retval, translation, quaternion, _ = solve_pnp_ransac(joints_14_rob_gt[b], uv_pred_np[b], intrinsic_np[b])
                        
#                        if pnp_retval is not False:
#                            print("uv_pred_np_b", uv_pred_np[b])
#                            print("joints_14_rob_gt[b]", joints_14_rob_gt[b])
#                            print("intrinsic_np[b]", intrinsic_np[b])
#                            print("pnp_retval", pnp_retval)
#                            print("translation", translation)
#                            print("quaternion", quaternion)
                        
                        this_pose_pred = np.eye(4)
                        this_pose_pred[:3, :3] = quaternion.matrix33.tolist()
                        this_pose_pred[:3, -1] = translation
                        pose_pred[b] = this_pose_pred
                    
                    uv_pred_list.append(uv_pred)
                    uv_gt_list.append(uv_gt)
                    uv_pck = batch_pck_from_pose(uv_gt.detach().cpu().numpy(), uv_pred.detach().cpu().numpy())
                    uv_kps_pck = torch.linalg.norm((uv_gt-uv_pred), dim=-1) # B x N
                    uv_kps_pck_list.append(uv_kps_pck.detach().cpu().numpy().tolist())
                    uv_pck_list.append(uv_pck)
                    pose_pred = torch.from_numpy(pose_pred).float().to(device)
                    
                
                
                if gt_angle_flag:
                    joints_angle_pred = joints_1d_gt[..., None]  
                
                pose_pred_lst.append(pose_pred)
                pose_gt_lst.append(pose_gt) 
 
                joints_kps_3d_pred = compute_kps_joints_loss(pose_pred[:, :3, :3], (pose_pred[:, :3, 3])[:, :, None], joints_angle_pred, device)  

                kps_add = kps_add + batch_add_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy())
                kps_mAP.append(batch_mAP_from_pose(joints_kps_3d_pred.detach().cpu().numpy(), joints_kps_3d_gt.detach().cpu().numpy(), thresholds))
                
                kps_pred_lst.append(joints_kps_3d_pred)
                kps_gt_lst.append(joints_kps_3d_gt)

                
            #torch.distributed.barrier()
            kps_mAP = np.concatenate(kps_mAP, axis=0).tolist()
            uv_pck_list = np.concatenate(uv_pck_list, axis=0) 
                        
#            print(uv_pred_list)
#            if pred_2d_flag:
#                uv_pck_list = np.concatenate(uv_pck_list, axis=0) 
#                uv_pck_list = distributed_concat(torch.from_numpy(uv_pck_list).to(device), len(sampler.dataset))
#                uv_kps_pck_list = np.concatenate(uv_kps_pck_list, axis=0) 
#                uv_kps_pck_list = distributed_concat(torch.from_numpy(uv_kps_pck_list).to(device), len(sampler.dataset))
#                uv_pred_list = distributed_concat(torch.cat(uv_pred_list,dim=0), len(sampler.dataset))
#                uv_gt_list = distributed_concat(torch.cat(uv_gt_list,dim=0), len(sampler.dataset))
                
#            kps_add  = distributed_concat(torch.from_numpy(np.array(kps_add)).to(device), len(sampler.dataset))
#            kps_mAP = distributed_concat(torch.from_numpy(np.array(kps_mAP)).to(device), len(sampler.dataset))
#            
#            pose_pred_gather = distributed_concat(torch.cat(pose_pred_lst, dim=0), len(sampler.dataset))
#            pose_gt_gather = distributed_concat(torch.cat(pose_gt_lst, dim=0), len(sampler.dataset))
#            
#            kps_pred_gather = distributed_concat(torch.cat(kps_pred_lst, dim=0), len(sampler.dataset))
#            kps_gt_gather = distributed_concat(torch.cat(kps_gt_lst, dim=0), len(sampler.dataset))
            
#            kps_add = kps_add.tolist()
#            kps_mAP = kps_mAP.tolist()
#            #pose_gt_gather = pose_gt_gather.tolist()
#            if pred_2d_flag:
#                uv_pred_list = uv_pred_list.detach().cpu().numpy()
#                uv_gt_list = uv_gt_list.detach().cpu().numpy()
#                uv_pck_list = uv_pck_list.detach().cpu().numpy()
#                uv_kps_pck_list = uv_kps_pck_list.detach().cpu().numpy()
#                pck_results = pck_metrics(uv_pck_list)
            
            kps_add_results = add_metrics(kps_add, add_thresh)
            kps_mAP_results = np.round(np.mean(kps_mAP, axis=0) * 100, 2)
            pck_results = pck_metrics(np.array(uv_pck_list))
            kps_mAP_dict = dict()
            
#            change_intrinsic_flag = cfg["CHANGE_INTRINSIC"]
            real_name = dataset_cfg["REAL_ROOT"].split('/')[-2]
#            #real_name = dataset_cfg["VAL_ROOT"].split('/')[-2]
            file_name = os.path.join(results_path, f"Epoch_{str(start_epoch).zfill(5)}_{real_name}_angle_{gt_angle_flag}_pred2d_{pred_2d_flag}_pnp.txt")
#            path_meta = os.path.join(info_path, f"Epoch_{str(start_epoch).zfill(5)}_{real_name}_change_intrin_{change_intrinsic_flag}_angle_{gt_angle_flag}_ns_{num_samples}_pred2d_{pred_2d_flag}_{method}.json")
#            
#            file_write_meta = open(path_meta, 'w')
#            #print("joints_3d_pred_gather", joints_3d_pred_repeat_gather.shape)
#            if dist.get_rank() == 0:
#                print("depth_path_lst", len(depth_path_lst))
#                print("pose_gt_lst", np.array(pose_gt_gather).shape)
#                if pred_2d_flag:
#                    print("uv_pred_list", uv_pred_list.shape)
#                    print("uv_gt_list", uv_gt_list.shape)
#                    print("uv_pck_list", uv_pck_list.shape)
#                    print("uv_kps_pck_list", uv_kps_pck_list.shape)
            
            
#            if dist.get_rank() == 0:    
#                print("percentage", np.percentile(uv_pred_list, [i * 10 for i in range(1, 10)]))            
            with open(file_name, "w") as f:
                print_to_screen_and_file(
                f, "Analysis results for dataset: {}".format(split[mode][1])
                )
                print_to_screen_and_file(
                f, "Number of frames in this dataset: {}".format(len(ass_add))
                )
                print_to_screen_and_file(f, "")
                
                
                print_to_screen_and_file(
                    f, f" uv_kps_pck : {np.mean(uv_kps_pck_list, axis=0)}")
                                    
                                    
                # print mAP
                for thresh, avg_map in zip(thresholds, kps_mAP_results):
                    print_to_screen_and_file(
                    f, " acc thresh: {:.5f} m".format(thresh)
                    )
                    print_to_screen_and_file(
                    f, " acc: {:.5f} %".format(float(avg_map))
                    )
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












