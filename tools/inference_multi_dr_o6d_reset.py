import time
import multiprocessing as mp
import os
import numpy as np
from tabulate import tabulate
import torch
import cv2
from PIL import Image as PILImage
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn import functional as F
from torchvision.utils import make_grid
from depth_c2rp.utils.utils import save_model, load_model, exists_or_mkdir, visualize_training_loss, find_seq_data_in_dir, load_camera_intrinsics
from depth_c2rp.utils.utils import batch_quaternion_matrix, compute_concat_loss,exists_or_mkdir, get_K_crop_resize, update_translation, compute_rotation_matrix_from_ortho6d, get_meshes_center, set_random_seed
from depth_c2rp.utils.image_proc import get_nrm
from depth_c2rp.utils.image_proc import batch_resize_masks_inference,depth_to_xyz
from depth_c2rp.utils.analysis import add_from_pose, add_metrics, print_to_screen_and_file, batch_add_from_pose, batch_mAP_from_pose, batch_acc_from_joint_angles
from depth_c2rp.datasets.datasets_o6d import Depth_dataset
from depth_c2rp.configs.config import update_config
from depth_c2rp.build import build_model
from depth_c2rp.optimizers import get_optimizer, adapt_lr
from depth_c2rp.DifferentiableRenderer.Kaolin.Renderer import DiffPFDepthRenderer

def network_inference(model, cfg, epoch_id, device, mAP_thresh=[0.02, 0.11, 0.01], add_thresh=0.1,angles_thresh=[2.5, 30.0, 2.5]):
    #set_random_seed(int(cfg["MODEL"]["SEED"]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    model.eval()

    # Build Testing Dataloader
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    testing_data_dir = dataset_cfg["TESTING_ROOT"]
    dr_engine = cfg["DR"]["ENGINE"]
    eval_cfg = cfg["EVAL"]
    testing_data = find_seq_data_in_dir(testing_data_dir)
    is_res = dataset_cfg["IS_RES"]
    camera_K = load_camera_intrinsics(os.path.join(dataset_cfg["TRAINING_ROOT"], "_camera_settings.json"))
    testing_dataset = Depth_dataset(testing_data, dataset_cfg["MANIPULATOR"], dataset_cfg["KEYPOINT_NAMES"], dataset_cfg["JOINT_NAMES"], \
    dataset_cfg["INPUT_RESOLUTION"], mask_dict=dataset_cfg["MASK_DICT"], camera_K = camera_K, is_res = dataset_cfg["IS_RES"], device=device)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=eval_cfg["BATCH_SIZE"], shuffle=True, \
                                               num_workers=int(eval_cfg["NUM_WORKERS"]), pin_memory=True, drop_last=False)
    
    img_h, img_w = train_cfg["IMAGE_SIZE"]
    img_size = (img_h, img_w)
    # Inference 
    ass_add = []
    ass_mAP = []
    ori_add = []
    ori_mAP = []
    angles_acc = []
    K = torch.tensor([
                    [502.30, 0.0, 319.5],
                    [0.0, 502.30, 179.5],
                    [0.0, 0.0, 1.0]
                    ], device=device)
    num_pts = cfg["DR"]["NUM_OF_SAMPLE_PTS"]
    start_thresh_mAP, end_thresh_mAP, interval_mAP = mAP_thresh
    thresholds = np.arange(start_thresh_mAP, end_thresh_mAP, interval_mAP)
    thresh_length = len(thresholds)
    
    start_angle_acc, end_angle_acc, interval_acc = angles_thresh
    acc_thresholds = np.arange(start_angle_acc, end_angle_acc, interval_acc)
    
    dr_iter_num = int(cfg["DR_ITER_NUM"])
    
    
    if dr_engine == "Kaolin":
        DPRenderer = DiffPFDepthRenderer(cfg, device)
        DPRenderer.load_mesh()
        DPRenderer.set_camera_intrinsics(K, width=img_w, height=img_h)
    
    for batch_idx, batch in enumerate(tqdm(test_loader)):
#        if batch_idx > 10:
#            break
        with torch.no_grad():
            start_time = time.time()
            next_img = batch["next_frame_img_as_input"].to(device)
            next_xy_wrt_cam, next_uv, next_simdepth = batch["next_frame_xy_wrt_cam"].to(device), batch["next_frame_uv"].to(device), \
            batch["next_frame_simdepth_as_input"].to(device)
            next_whole_simdepth = batch["next_frame_whole_simdepth_as_input"].to(device)
            next_simdetph_all = batch["next_simdepth_all"].to(device)
            
            batch_gt_quaternion, batch_gt_trans, batch_gt_joints_wrt_cam, batch_gt_joints_wrt_rob = batch["next_frame_base_quaternion"].to(device), batch["next_frame_base_trans"].to(device), batch["next_frame_joints_wrt_cam"].to(device), batch["next_frame_joints_wrt_rob"].to(device) # N x k
            batch_gt_rot = batch_quaternion_matrix(batch_gt_quaternion.T)
            batch_gt_K, batch_gt_boxes = batch["next_camera_K"].to(device), batch["next_bbox"].to(device)
            batch_gt_joints_pos = batch["next_frame_joints_pos"].to(device)
            next_normals = get_nrm(next_simdetph_all, K[0][0], batch_gt_boxes[0], dataset_cfg["INPUT_RESOLUTION"])
            
            
            next_input = torch.cat([next_img, next_simdepth, next_xy_wrt_cam, next_uv, next_normals], dim=1)
            batch_dt_mask, batch_dt_poses, batch_dt_joints_pos = model(next_input) 
            
            index = DPRenderer.get_sample_index(num_pts)
            batch_sample_gt_wrt_rob = DPRenderer.get_sample_meshes(batch_gt_joints_pos, index) # B x num_pts x 3
            
            
            batch_gt_crop_resize = cfg["DATASET"]["INPUT_RESOLUTION"]
            batch_new_gt_K = get_K_crop_resize(batch_gt_K, batch_gt_boxes, batch_gt_crop_resize)
            batch_t_init = get_meshes_center(batch_sample_gt_wrt_rob) # B x 3
            batch_t_init = (torch.bmm(batch_gt_rot, batch_t_init[:, :, None]) + batch_gt_trans[:, :, None]).squeeze(2)
            batch_dt_trans = update_translation(batch_dt_poses[..., 6:], batch_new_gt_K, batch_t_init)
            
            batch_dt_o6dposes = batch_dt_poses[..., :6]
            
            
            end_time = time.time()
        #print("all_time", end_time - start_time)
        
        if dr_engine == "Kaolin" and dr_iter_num > 0:
            batch_dt_mask = batch_resize_masks_inference(batch_dt_mask, img_size)
            batch_dt_trans.requires_grad = True
            batch_dt_o6dposes.requires_grad = True
            batch_dt_joints_pos.requires_grad = True
            # batch_dt_quaternion : B x 4
            # batch_dt_trans : B x 3
            # batch_dt_joints_pos : B x 8 x 1
            # batch_dt_mask : B x 1 x img_h x img_w
            # print("next_frame_whole_simdepth_as_input", next_whole_simdepth.shape)
            DPRenderer.set_o6dpose_optimizer(batch_dt_o6dposes, batch_dt_trans, batch_dt_joints_pos)
#            print("batch_gt_base_trans", batch_gt_base_trans)
#            DPRenderer.set_optimizer(batch_gt_base_quaternion, batch_gt_base_trans, batch_gt_joints_pos)
#            print("next_frame_img_path" , batch["next_frame_img_path"])
#            #np.savetxt(f"./gt_cam.txt", batch_gt_joints_wrt_cam.reshape(-1,3).detach().cpu().numpy())
#            print("next_whole_simdepth", next_whole_simdepth.shape) # B x H x W x 1
#            next_whole_xyz = depth_to_xyz(next_whole_simdepth.permute(0,3,1,2),f=502.30) # B x 3 x H x W
#            for b in range(next_whole_simdepth.shape[0]):
#                cv2.imwrite(f"./check_depths/gt_{b}_depth.png", (next_whole_simdepth[b]*255).detach().cpu().numpy())
#                np.savetxt(f"./check_depths/gt_{b}_xyz.txt", next_whole_xyz[b].permute(1,2,0).reshape(-1,3).detach().cpu().numpy())
            
#            save_path = f"./check_depths/concat_imgs"
#            exists_or_mkdir(save_path)
            
            B, H, W, _ = next_whole_simdepth.shape
            all_res = np.zeros((B, 3, H, dr_iter_num * W))
            
            DPRenderer.batch_mesh(batch_dt_mask.shape[0])
            for update_idx in range(dr_iter_num):
                DPRenderer.GA_optimizer_zero_grad()
                DPRenderer.RT_optimizer_zero_grad()
                
                DPRenderer.concat_mesh(rot_type=cfg["MODEL"]["ROT_TYPE"])
                DPRenderer.Rasterize()
                DPRenderer.loss_forward(next_whole_simdepth, batch_dt_mask, img_path=batch["next_frame_img_path"], update_idx=update_idx)
#                all_res[:, :, :, update_idx * W : (update_idx+1) * W] = res.transpose(0, 3, 1, 2)
                DPRenderer.loss_backward()
                
                DPRenderer.RT_optimizer_step()
                DPRenderer.GA_optimizer_step()
                
#                print("batch_dt_quaternion", batch_dt_quaternion.grad)
#                print("batch_dt_trans", batch_dt_trans.grad)
#                print("batch_dt_joints_pos", batch_dt_joints_pos.grad)
                
        
#        grid_image = make_grid(torch.from_numpy(all_res), 1, normalize=False, scale_each=False) 
##        print("grid_image.shape", grid_image.shape) 
##        grid_image = PILImage.fromarray((grid_image.detach().cpu().numpy()))
##        grid_image.save(os.path.join(save_path, f"blend.png"))
##        grid_image = grid_image.detach().cpu().numpy()
##        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
#        cv2.imwrite(os.path.join(save_path, f"blend.png"), grid_image.detach().cpu().numpy().transpose(1,2,0)[:,:,::-1])
#        
#        break                
            
        
        
        batch_dt_rot = compute_rotation_matrix_from_ortho6d(batch_dt_o6dposes)
        
        
        batch_dt_joints_wrt_cam_ass = compute_concat_loss(batch_dt_rot, batch_dt_trans[:, :, None], batch_dt_joints_pos, batch_dt_rot.device)
        batch_dt_joints_wrt_cam_ori = (torch.bmm(batch_dt_rot, batch_gt_joints_wrt_rob.permute(0, 2, 1).contiguous()) + batch_dt_trans[:, :, None]).permute(0,2,1) 
        
        
        batch_dt_joints_pos = batch_dt_joints_pos.detach().cpu().numpy()
        batch_gt_joints_pos = batch_gt_joints_pos.detach().cpu().numpy()
        batch_dt_joints_wrt_cam_ass = batch_dt_joints_wrt_cam_ass.detach().cpu().numpy()
        batch_dt_joints_wrt_cam_ori = batch_dt_joints_wrt_cam_ori.detach().cpu().numpy()
        batch_gt_joints_wrt_cam = batch_gt_joints_wrt_cam.detach().cpu().numpy()
        batch_gt_joints_wrt_rob = batch_gt_joints_wrt_rob.detach().cpu().numpy()
        batch_xyz_rp = batch["next_xyz_rp"].numpy()
        #print("test_batch_xyz_rp", batch_xyz_rp)
        batch_size = batch_gt_quaternion.shape[0]
        
        ass_add_mean = batch_add_from_pose(batch_dt_joints_wrt_cam_ass, batch_gt_joints_wrt_cam) # list of size B
        ass_add = ass_add + ass_add_mean
        
        ass_mAP_mean = batch_mAP_from_pose(batch_dt_joints_wrt_cam_ass, batch_gt_joints_wrt_cam,thresholds) # 
        ass_mAP.append(ass_mAP_mean)
        
        ori_add_mean = batch_add_from_pose(batch_dt_joints_wrt_cam_ori, batch_gt_joints_wrt_cam) # list of size B
        ori_add = ori_add + ori_add_mean
        
        ori_mAP_mean = batch_mAP_from_pose(batch_dt_joints_wrt_cam_ori, batch_gt_joints_wrt_cam,thresholds) # 
        ori_mAP.append(ori_mAP_mean)
        #print(np.array(mAP).shape)
        
        
        angles_acc_mean = batch_acc_from_joint_angles(batch_dt_joints_pos, batch_gt_joints_pos, acc_thresholds) # 
        angles_acc.append(angles_acc_mean)
            
            
    #print("thresh_length", thresh_length)
    ass_add_results = add_metrics(np.array(ass_add), add_thresh)
    ass_mAP_results = np.round(np.mean(ass_mAP, axis=0) * 100, 2)
    ass_mAP_dict = dict()
    ori_add_results = add_metrics(np.array(ori_add), add_thresh)
    ori_mAP_results = np.round(np.mean(ori_mAP, axis=0) * 100, 2)
    ori_mAP_dict = dict()
    angles_results = np.round(np.mean(angles_acc,axis=0)*100, 2)
    angles_dict = dict()
    
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
        
        # print add
        print_to_screen_and_file(
            f, " ADD AUC: {:.5f}".format(ori_add_results["add_auc"])
        )
        print_to_screen_and_file(
            f,
               " ADD  AUC threshold: {:.5f} m".format(ori_add_results["add_auc_thresh"]),
        )
        print_to_screen_and_file(
            f, " ADD  Mean: {:.5f}".format(ori_add_results["add_mean"])
        )
        print_to_screen_and_file(
            f, " ADD  Median: {:.5f}".format(ori_add_results["add_median"])
        )
        print_to_screen_and_file(
            f, " ADD  Std Dev: {:.5f}".format(ori_add_results["add_std"]))
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
        for thresh, avg_map in zip(thresholds, ori_mAP_results):
            print_to_screen_and_file(
            f, " acc thresh: {:.5f} m".format(thresh)
            )
            print_to_screen_and_file(
            f, " acc: {:.5f} %".format(float(avg_map))
            )
            ori_mAP_dict[str(thresh)] = float(avg_map)
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
            
    return ass_add_results, ass_mAP_dict, ori_add_results, ori_mAP_dict, angles_dict
                



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg, args = update_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    model_cfg = cfg["MODEL"]
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    model = build_model(model_cfg["BACKBONE"], model_cfg["HEAD"], model_cfg["MODEL_CLASSES"], model_cfg["IN_CHANNELS"], \
                        dataset_cfg["NUM_JOINTS"], dataset_cfg["OUTPUT_RESOLUTION"][0], dataset_cfg["OUTPUT_RESOLUTION"][1],model_cfg["ROT_TYPE"])
    model.init_pretrained(model_cfg["PRETRAINED"])
    
    optim_cfg = cfg["OPTIMIZER"]
    #optimizer = get_optimizer(model, optim_cfg["NAME"], optim_cfg["LR"], optim_cfg["WEIGHT_DECAY"])
    epoch_id = cfg["EPOCH_ID"]
    exp_id = cfg["EXP_ID"]
    this_ckpt_path = f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/{exp_id}/CHECKPOINT/model_{str(epoch_id).zfill(3)}.pth"
    model, start_epoch = load_model(model, this_ckpt_path, optim_cfg["LR"])
    model = model.to(device)
    network_inference(model, cfg, epoch_id, device)













