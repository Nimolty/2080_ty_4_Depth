import time
import multiprocessing as mp
import os
import numpy as np
from tabulate import tabulate
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn import functional as F
 
from depth_c2rp.utils.utils import save_model, load_model, exists_or_mkdir, visualize_training_loss, find_seq_data_in_dir, load_camera_intrinsics
from depth_c2rp.utils.utils import batch_quaternion_matrix, compute_concat_loss
from depth_c2rp.utils.analysis import add_from_pose, add_metrics, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose
from depth_c2rp.datasets.datasets import Depth_dataset
from depth_c2rp.configs.config import update_config
from depth_c2rp.build import build_model
from depth_c2rp.optimizers import get_optimizer, adapt_lr
def network_inference(model, cfg, epoch_id, device, mAP_thresh=[0.02, 0.11, 0.01], add_thresh=0.1):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    model.eval()

    # Build Testing Dataloader
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    testing_data_dir = dataset_cfg["TESTING_ROOT"]
    testing_data = find_seq_data_in_dir(testing_data_dir)
    is_res = dataset_cfg["IS_RES"]
    camera_K = load_camera_intrinsics(os.path.join(dataset_cfg["TRAINING_ROOT"], "_camera_settings.json"))
    testing_dataset = Depth_dataset(testing_data, dataset_cfg["MANIPULATOR"], dataset_cfg["KEYPOINT_NAMES"], dataset_cfg["JOINT_NAMES"], \
    dataset_cfg["INPUT_RESOLUTION"], mask_dict=dataset_cfg["MASK_DICT"], camera_K = camera_K, is_res = dataset_cfg["IS_RES"], device=device)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=train_cfg["BATCH_SIZE"], shuffle=True, \
                                               num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True)
                                               
    # Inference 
    with torch.no_grad():
        # network output
        add = []
        mAP = []
        
        start_thresh_mAP, end_thresh_mAP, interval_mAP = mAP_thresh
        thresholds = np.arange(start_thresh_mAP, end_thresh_mAP, interval_mAP)
        thresh_length = len(thresholds)
        
        
        for batch_idx, batch in enumerate(tqdm(test_loader)):
#            if batch_idx > 10:
#                break
            start_time = time.time()
            next_img = batch["next_frame_img_as_input"].to(device)
            next_xy_wrt_cam, next_uv, next_simdepth, next_normals = batch["next_frame_xy_wrt_cam"].to(device), batch["next_frame_uv"].to(device), \
            batch["next_frame_simdepth_as_input"].to(device), batch["next_normals_crop"].to(device)
            next_input = torch.cat([next_img, next_simdepth, next_xy_wrt_cam, next_uv, next_normals], dim=1)
            
            
            batch_gt_base_quaternion, batch_gt_base_trans, batch_gt_joints_wrt_cam, batch_gt_joints_wrt_rob = batch["next_frame_base_quaternion"].to(device), batch["next_frame_base_trans"].to(device), batch["next_frame_joints_wrt_cam"].to(device), batch["next_frame_joints_wrt_rob"].to(device) # N x k
            batch_dt_mask, batch_dt_trans, batch_dt_quaternion, batch_dt_joints_pos = model(next_input) 
            
            batch_dt_quaternion_norm = batch_dt_quaternion / torch.norm(batch_dt_quaternion, dim=-1,keepdim=True)
            batch_rot = batch_quaternion_matrix(batch_dt_quaternion_norm.T)
            batch_dt_joints_wrt_cam = compute_concat_loss(batch_rot, batch_dt_trans[:, :, None], batch_dt_joints_pos, batch_rot.device)
            
            end_time = time.time()
            #print("all_time", end_time - start_time)
            
            batch_dt_trans = batch_dt_trans.detach().cpu().numpy()
            batch_dt_quaternion = batch_dt_quaternion.detach().cpu().numpy()
            batch_dt_joints_pos = batch_dt_joints_pos.detach().cpu().numpy()
            batch_dt_joints_wrt_cam = batch_dt_joints_wrt_cam.detach().cpu().numpy()
            batch_gt_joints_wrt_cam = batch_gt_joints_wrt_cam.detach().cpu().numpy()
            batch_gt_joints_wrt_rob = batch_gt_joints_wrt_rob.detach().cpu().numpy()
            batch_xyz_rp = batch["next_xyz_rp"].numpy()
            #print("test_batch_xyz_rp", batch_xyz_rp)
            batch_size = batch_gt_base_quaternion.shape[0]
            
            add_mean = batch_add_from_pose(batch_dt_joints_wrt_cam, batch_gt_joints_wrt_cam) # list of size B
            add = add + add_mean
            
            mAP_mean = batch_mAP_from_pose(batch_dt_joints_wrt_cam, batch_gt_joints_wrt_cam,thresholds) # list of M x B
            mAP = mAP + mAP_mean
 
                
                
        print("thresh_length", thresh_length)
        add_results = add_metrics(np.array(add), add_thresh)
        mAP_np = np.array(mAP).reshape(thresh_length, -1)
        mAP_results = np.round(np.mean(mAP, axis=1) * 100, 2)
        mAP_dict = dict()
        
        # Print File and Save Results
        save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
        results_path = os.path.join(save_path, "NUMERICAL_RESULTS")
        exists_or_mkdir(save_path)
        exists_or_mkdir(results_path)
        exp_id = cfg["EXP_ID"]
        file_name = os.path.join(results_path, f"EXP{str(exp_id).zfill(2)}_{str(epoch_id).zfill(3)}.txt")
        
        with open(file_name, "w") as f:
            print_to_screen_and_file(
            f, "Analysis results for dataset: {}".format(testing_data_dir)
            )
            print_to_screen_and_file(
            f, "Number of frames in this dataset: {}".format(len(add))
            )
            print_to_screen_and_file(f, "")
            
            # print add
            print_to_screen_and_file(
                f, " ADD AUC: {:.5f}".format(add_results["add_auc"])
            )
            print_to_screen_and_file(
                f,
                   " ADD  AUC threshold: {:.5f} m".format(add_results["add_auc_thresh"]),
            )
            print_to_screen_and_file(
                f, " ADD  Mean: {:.5f}".format(add_results["add_mean"])
            )
            print_to_screen_and_file(
                f, " ADD  Median: {:.5f}".format(add_results["add_median"])
            )
            print_to_screen_and_file(
                f, " ADD  Std Dev: {:.5f}".format(add_results["add_std"]))
            print_to_screen_and_file(f, "")
            
            # print mAP
            for thresh, avg_map in zip(thresholds, mAP_results):
                print_to_screen_and_file(
                f, " mAP thresh: {:.5f} m".format(thresh)
                )
                print_to_screen_and_file(
                f, " mAP: {:.5f} %".format(float(avg_map))
                )
                mAP_dict[str(thresh)] = float(avg_map)
            print_to_screen_and_file(f, "")
        return add_results, mAP_dict
                



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg, args = update_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    model_cfg = cfg["MODEL"]
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    model = build_model(model_cfg["BACKBONE"], model_cfg["HEAD"], model_cfg["MODEL_CLASSES"], model_cfg["IN_CHANNELS"], \
                        dataset_cfg["NUM_JOINTS"], dataset_cfg["OUTPUT_RESOLUTION"][0], dataset_cfg["OUTPUT_RESOLUTION"][1])
    model.init_pretrained(model_cfg["PRETRAINED"])
    
    optim_cfg = cfg["OPTIMIZER"]
    optimizer = get_optimizer(model, optim_cfg["NAME"], optim_cfg["LR"], optim_cfg["WEIGHT_DECAY"])
    
    this_ckpt_path = "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/20/CHECKPOINT/model.pth"
    model, optimizer, start_epoch = load_model(model, this_ckpt_path, optim_cfg["LR"], optimizer)
    model = model.to(device)
    network_inference(model, cfg, 100, device)













