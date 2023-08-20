import numpy as np
import torch
import open3d as o3d
import json
import os
import glob
import argparse
from tqdm import tqdm
from depth_c2rp.diffusion_utils.diffusion_o3d_visualize import visualize_dynamic_pts
from depth_c2rp.diffusion_utils.diffusion_2d_visualize import save_2d_pred_visual

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def get_2d_info(info, selected_idx, device):
    _2d_uv_preds = torch.from_numpy(np.array(info["uv_pred_list"])).float().to(device)
    _2d_uv_gts = torch.from_numpy(np.array(info["uv_gt_list"])).float().to(device)
    _path_lists = np.array(info["depth_path_lst"])
    _pcks = np.array(info["uv_pck_list"])
    _idx = np.argsort(_pcks)
    
    _idx = np.concatenate([_idx[:selected_idx], _idx[-selected_idx:]], axis=0)
    
    _2d_uv_pred = _2d_uv_preds[_idx]
    _2d_uv_gt = _2d_uv_gts[_idx]
    _path_list = _path_lists[_idx]
    print("pck", _pcks[_idx])
    return _2d_uv_pred, _2d_uv_gt, _path_list

#def get_3d_info(info, selected_idx, selected_all_idx, device):
#    _3d_pred_samples = torch.from_numpy(np.array(info["joints_3d_pred_samples"])).float().to(device) # B x num_samples x steps x N x 3
#    _3d_gts = torch.from_numpy(np.array(info["joints_3d_gt"])).float().to(device) # B x N x 3
#    _3d_poses = torch.from_numpy(np.array(info["pose_gt_lst"])).float().to(device) # B x 3 x 4
#    _path_lists = np.array(info["depth_path_lst"]) # lens : B
#    _adds = np.array(info["ass_add"]) # (B, )        
#    _3d_preds = torch.from_numpy(np.array(info["joints_3d_pred"])).float().to(device) # B x N x num_samples x 3
#    
#    _idx = np.argsort(_adds)
#    _idx_good = np.random.choice(_idx[:selected_all_idx], selected_idx)
#    _idx_bad = np.random.choice(_idx[-selected_all_idx:], selected_idx)
#    _idx = np.concatenate([_idx_good, _idx_bad], axis=0)
#    
#    _3d_gt = _3d_gts[_idx] # s x N x 3
#    _c2r_rot_tensors = _3d_poses[_idx][:, :3, :3] # s x 3 x 3
#    _c2r_trans_tensors = _3d_poses[_idx][:, :3, 3:] # s x 3 x 1
#    _path_list = _path_lists[_idx] # s
#    _3d_pred_sample = _3d_pred_samples[_idx] # s x num_samples x steps x N x 3
#    _3d_pred = _3d_preds[_idx]
#    
#    return _3d_gt, _c2r_rot_tensors, _c2r_trans_tensors, _path_list, _3d_pred_sample, _3d_pred.permute(0, 2, 1, 3)
    
def get_3d_info(info, selected_idx, device, random_flag=True):
    _3d_pred_samples = torch.from_numpy(np.array(info["joints_3d_pred_samples"])).float().to(device) # B x num_samples x steps x N x 3
    _3d_gts = torch.from_numpy(np.array(info["joints_3d_gt"])).float().to(device) # B x N x 3
    _3d_poses = torch.from_numpy(np.array(info["pose_gt_lst"])).float().to(device) # B x 3 x 4
    _path_lists = np.array(info["depth_path_lst"]) # lens : B
    _adds = np.array(info["ass_add"]) # (B, )        
    _3d_preds = torch.from_numpy(np.array(info["joints_3d_pred"])).float().to(device) # B x N x num_samples x 3
    
    if random_flag:
        _idx = np.argsort(_adds)
        _idx_good = np.random.choice(_idx[:selected_all_idx], selected_idx)
        _idx_bad = np.random.choice(_idx[-selected_all_idx:], selected_idx)
        _idx = np.concatenate([_idx_good, _idx_bad], axis=0)
    else:
        _idx = np.arange(selected_idx)
    
    _3d_gt = _3d_gts[_idx] # s x N x 3
    _c2r_rot_tensors = _3d_poses[_idx][:, :3, :3] # s x 3 x 3
    _c2r_trans_tensors = _3d_poses[_idx][:, :3, 3:] # s x 3 x 1
    _path_list = _path_lists[_idx] # s
    _3d_pred_sample = _3d_pred_samples[_idx] # s x num_samples x steps x N x 3
    _3d_pred = _3d_preds[_idx]
    
    return _3d_gt, _c2r_rot_tensors, _c2r_trans_tensors, _path_list, _3d_pred_sample, _3d_pred.permute(0, 2, 1, 3)
    

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info_unchange_paths = glob.glob(os.path.join(args.info_path, "*change_intrin_False_angle_False_ns_10_pred2d_True*"))
    info_change_paths = glob.glob(os.path.join(args.info_path, "*change_intrin_True_angle_False_ns_10_pred2d_True*"))
    
    info_unchange_paths.sort()
    info_change_paths.sort()
    print(info_change_paths)
    for info_unchange_path, info_change_path in tqdm(zip(info_unchange_paths[args.dataset_start_id : args.dataset_start_id + args.dataset_seq_id], info_change_paths[args.dataset_start_id : args.dataset_start_id + args.dataset_seq_id])):
        info_unchange = json.load(open(info_unchange_path, "r"))
        info_change = json.load(open(info_change_path, "r"))
        assert len(info_unchange["uv_pck_list"]) == len(info_change["uv_pck_list"])
        
        # 2D
        # unchange
        unchange_good_save_dir = os.path.join(args.save_path, "unchange", "good")
        unchange_bad_save_dir = os.path.join(args.save_path, "unchange", "bad")
        change_good_save_dir = os.path.join(args.save_path, "change", "good")
        change_bad_save_dir = os.path.join(args.save_path, "change", "bad")
        unchange_2d_uv_pred, unchange_2d_uv_gt, unchange_path_list = get_2d_info(info_unchange, args.pck_selected_idx, device)
        save_2d_pred_visual(unchange_2d_uv_pred[:args.pck_selected_idx], 
                            unchange_2d_uv_gt[:args.pck_selected_idx], 
                            unchange_path_list[:args.pck_selected_idx], 
                            unchange_good_save_dir,
                            change_intrin_flag=False,)
        save_2d_pred_visual(unchange_2d_uv_pred[args.pck_selected_idx:], 
                            unchange_2d_uv_gt[args.pck_selected_idx:], 
                            unchange_path_list[args.pck_selected_idx:], 
                            unchange_bad_save_dir,
                            change_intrin_flag=False)
        # change
        change_2d_uv_pred, change_2d_uv_gt, change_path_list = get_2d_info(info_change, args.pck_selected_idx, device)
        save_2d_pred_visual(change_2d_uv_pred[:args.pck_selected_idx], 
                            change_2d_uv_gt[:args.pck_selected_idx], 
                            change_path_list[:args.pck_selected_idx], 
                            change_good_save_dir,
                            change_intrin_flag=True,)
        save_2d_pred_visual(change_2d_uv_pred[args.pck_selected_idx:], 
                            change_2d_uv_gt[args.pck_selected_idx:], 
                            change_path_list[args.pck_selected_idx:], 
                            change_bad_save_dir,
                            change_intrin_flag=True)
        
        # 3D first
        # unchange
        unchange_3d_gt, unchange_c2r_rot_tensors, unchange_c2r_trans_tensors, unchange_path_list, unchange_3d_pred_sample, unchange_3d_pred = get_3d_info(info_unchange, args.selected_idx, args.selected_all_idx, device)
        
        # change
        change_3d_gt, change_c2r_rot_tensors, change_c2r_trans_tensors, change_path_list, change_3d_pred_sample, change_3d_pred = get_3d_info(info_change, args.selected_idx, args.selected_all_idx, device)
        
        print(unchange_3d_pred_sample.shape)
        
        for s in tqdm(range(unchange_3d_pred_sample.shape[0])):
            unchange_path = unchange_path_list[s]
            change_path = change_path_list[s]
            unchange_cat_id, unchange_frame_id = unchange_path.split('/')[-2], unchange_path.split('/')[-1][:6]
            change_cat_id, change_frame_id = change_path.split('/')[-2], change_path.split('/')[-1][:6]
            
            if s < unchange_3d_pred_sample.shape[0]//2:
                unchange_save_dir = os.path.join(args.save_path, "unchange", "good3d", unchange_cat_id, unchange_frame_id)
                change_save_dir = os.path.join(args.save_path, "change", "good3d", change_cat_id, change_frame_id) 
            else:  
                unchange_save_dir = os.path.join(args.save_path, "unchange", "bad3d", unchange_cat_id, unchange_frame_id)    
                change_save_dir = os.path.join(args.save_path, "change", "bad3d", change_cat_id, change_frame_id)    
            for n in range(unchange_3d_pred_sample.shape[1]):
                this_unchange_save_dir = os.path.join(unchange_save_dir, f"{str(n).zfill(2)}")
                this_change_save_dir = os.path.join(change_save_dir, f"{str(n).zfill(2)}")                    
                visualize_dynamic_pts(unchange_3d_pred_sample[s][n], 
                                      unchange_3d_gt[s], 
                                      unchange_3d_pred[s][n],
                                      unchange_c2r_rot_tensors[s], 
                                      unchange_c2r_trans_tensors[s], 
                                      this_unchange_save_dir, 
                                      args.view_path,
                                      width=args.width,
                                      height=args.height,)
                
                visualize_dynamic_pts(change_3d_pred_sample[s][n], 
                                      change_3d_gt[s], 
                                      change_3d_pred[s][n],
                                      change_c2r_rot_tensors[s], 
                                      change_c2r_trans_tensors[s], 
                                      this_change_save_dir, 
                                      args.view_path,
                                      width= args.width,
                                      height=args.height,)
        

def main_unchange_trainval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info_unchange_paths = glob.glob(os.path.join(args.info_path, "*change_intrin_False_angle_False_ns_10_pred2d_False*"))
    
    info_unchange_paths.sort()
    print(info_unchange_paths)
    for info_unchange_path in tqdm(info_unchange_paths[args.dataset_start_id : args.dataset_start_id + args.dataset_seq_id]):
        info_unchange = json.load(open(info_unchange_path, "r"))
        
        info_unchange_epoch_id = info_unchange["diff epoch"]
        sampler_name = info_unchange["sampler name"]
        
        # 2D
        # unchange
#        unchange_good_save_dir = os.path.join(args.save_path, "unchange", "good")
#        unchange_bad_save_dir = os.path.join(args.save_path, "unchange", "bad")
#
#        unchange_2d_uv_pred, unchange_2d_uv_gt, unchange_path_list = get_2d_info(info_unchange, args.pck_selected_idx, device)
#        save_2d_pred_visual(unchange_2d_uv_pred[:args.pck_selected_idx], 
#                            unchange_2d_uv_gt[:args.pck_selected_idx], 
#                            unchange_path_list[:args.pck_selected_idx], 
#                            unchange_good_save_dir,
#                            change_intrin_flag=False,
#                            scale=1.0, 
#                            delta=0.0)
#        save_2d_pred_visual(unchange_2d_uv_pred[args.pck_selected_idx:], 
#                            unchange_2d_uv_gt[args.pck_selected_idx:], 
#                            unchange_path_list[args.pck_selected_idx:], 
#                            unchange_bad_save_dir,
#                            change_intrin_flag=False,
#                            scale=1.0, 
#                            delta=0.0)
        
        # 3D first
        # unchange
        unchange_3d_gt, unchange_c2r_rot_tensors, unchange_c2r_trans_tensors, unchange_path_list, unchange_3d_pred_sample, unchange_3d_pred = get_3d_info(info_unchange, args.selected_idx, device, random_flag=False)
                
        print(unchange_3d_pred_sample.shape)
        
        for s in tqdm(range(unchange_3d_pred_sample.shape[0])):
            unchange_path = unchange_path_list[s]
            unchange_frame_id = unchange_path.split('/')[-1][:8]
            
#            if s < unchange_3d_pred_sample.shape[0]//2:
#                unchange_save_dir = os.path.join(args.save_path, "unchange", "good3d", info_unchange_epoch_id, sampler_name, unchange_frame_id) 
#            else:  
#                unchange_save_dir = os.path.join(args.save_path, "unchange", "bad3d", info_unchange_epoch_id, sampler_name, unchange_frame_id) 
                
            unchange_save_dir = os.path.join(args.save_path, "unchange", "same", info_unchange_epoch_id, unchange_frame_id)       
                      
            for n in range(unchange_3d_pred_sample.shape[1]):
                this_unchange_save_dir = os.path.join(unchange_save_dir, f"{str(n).zfill(2)}") 
                if os.path.exists(this_unchange_save_dir):
                    continue            
                visualize_dynamic_pts(unchange_3d_pred_sample[s][n], 
                                      unchange_3d_gt[s], 
                                      unchange_3d_pred[s][n],
                                      unchange_c2r_rot_tensors[s], 
                                      unchange_c2r_trans_tensors[s], 
                                      this_unchange_save_dir, 
                                      args.view_path,
                                      width=args.width,
                                      height=args.height,)
                   





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_path", type=str, default="/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/DIFF/diff_26/INFERENCE_LOGS/ode")
    parser.add_argument("--save_path", type=str, default="/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/DIFF/diff_26/INFERENCE_LOGS/ode/")
    parser.add_argument("--view_path", type=str, default="/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/diffusion_utils/viewpoint.json")
    parser.add_argument("--selected_all_idx",type=int, default=80)
    parser.add_argument("--selected_idx", type=int, default=20)
    parser.add_argument("--pck_selected_idx", type=int, default=30)
    parser.add_argument("--dataset_start_id", type=int, default=0)
    parser.add_argument("--dataset_seq_id", type=int, default=2)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()
    main_unchange_trainval(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    