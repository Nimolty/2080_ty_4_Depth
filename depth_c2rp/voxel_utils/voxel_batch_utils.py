import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from depth_c2rp.voxel_utils.extensions.ray_aabb.jit import ray_aabb
from depth_c2rp.voxel_utils.extensions.pcl_aabb.jit import pcl_aabb

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, Dataset)
import torchvision.ops as tv_ops
import torch.nn as nn 

import numpy as np
import time
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import depth_c2rp.voxel_utils.constants as constants
import json
from tqdm import tqdm

from depth_c2rp.datasets.datasets_voxel_ours import Voxel_dataset
from depth_c2rp.voxel_utils.voxel_network import build_model, build_voxel_refine_network
from torch_scatter import scatter, scatter_softmax, scatter_max, scatter_log_softmax
from depth_c2rp.models.backbones.dream_hourglass import ResnetSimpleWoff
from depth_c2rp.voxel_utils.refine_batch_utils import get_pred_refine, compute_refine_loss
from depth_c2rp.configs.config import update_config


def DepthToPointCloud(depthImage, f):
    H, W, _ = depthImage.shape
    depthImage_copy = deepcopy(depthImage[:, :, 0])
    du = W//2 - 0.5
    dv = H//2 - 0.5

    pointCloud = np.zeros((H, W, 3)) #
    IndexX = np.arange(0, W)[None, :] - du
    IndexY = np.arange(0, H)[:, None] - dv
    
    pointCloud[:, :, 0] = depthImage_copy * IndexX / f
    pointCloud[:, :, 1] = depthImage_copy * IndexY / f
    pointCloud[:, :, 2] = depthImage_copy
    
    return pointCloud

def sample_valid_points(valid_mask, sample_num, block_x=8, block_y=8, data_dict=None):
    bs,h,w = valid_mask.shape
    assert h % block_y == 0
    assert w % block_x == 0
    # reshape valid mask to make sure non zero returns in the block order other than column order.
    valid_mask = valid_mask.reshape(bs,h//block_y,block_y,w).permute(0,1,3,2).contiguous()
    valid_mask = valid_mask.reshape(bs,h//block_y,w//block_x,block_x,block_y).permute(0,1,2,4,3).contiguous()
    valid_idx = torch.nonzero(valid_mask, as_tuple=False)
    valid_bid = valid_idx[:,0]
    # since nonzero return in c seq. we can make sure valid_bid is sorted
    # use torch.unique_consecutive to avoid sorting
    _, example_cnt = torch.unique_consecutive(valid_bid, return_counts=True)
    bid_interval = torch.cumsum(example_cnt,0)
    bid_interval = torch.cat((torch.Tensor([0]).long().to(valid_mask.device), bid_interval),0)
    # Now we use for loop over batch dim. can be accelerated by cuda kernal
    tmp_list = []
    #print(data_dict["path"])
    for i in range(bid_interval.shape[0]-1):
        sid = bid_interval[i]
        eid = bid_interval[i+1]
        cur_cnt = eid - sid
        if cur_cnt < sample_num:
            if data_dict is not None:
                print(data_dict["path"])
            print("bid_interval", bid_interval)
            print("sample_num", sample_num)
            print("cur_cnt", cur_cnt)
            print("eid", eid)
            print("sid", sid)
            mult = np.ceil(float(sample_num)/float(cur_cnt)) #- 1
            print("mult", mult)
            cur_points_idx = torch.arange(sid,eid).long().to(valid_mask.device)
            print("cur_pints_idx", cur_points_idx)
            rand_pool = cur_points_idx.repeat(int(mult))
            print("rand_pool", rand_pool)
            nextra = sample_num - cur_cnt
            print("nextra", nextra)
            rand_pool_idx = np.random.choice(rand_pool.shape[0], nextra.item(), replace=False)
            print("rand pool idx", rand_pool_idx)
            extra_idx = rand_pool[rand_pool_idx]
            print("extra_idx", extra_idx)
            sample_idx = torch.cat([cur_points_idx, extra_idx], dim=0)
        else:
            sample_step = cur_cnt // sample_num
            interval_num = cur_cnt // sample_step
            sample_offset = torch.randint(low=0,high=sample_step,size=(interval_num,)).to(valid_mask.device)
            sample_idx = sid + sample_offset + sample_step * torch.arange(interval_num).long().to(valid_mask.device)
            if sample_num <= sample_idx.shape[0]:
                tmp_idx = torch.randperm(sample_idx.shape[0])[:sample_num].long().to(valid_mask.device)
                sample_idx = sample_idx[tmp_idx]
            else:
                raise ValueError('Should be samller')
        
        tmp_list.append(valid_idx[sample_idx])
    sampled_valid_idx = torch.cat(tmp_list,0)
    sampled_flat_img_id = (sampled_valid_idx[:,1] * block_y + sampled_valid_idx[:,3]) * w \
                        + sampled_valid_idx[:,2] * block_x + sampled_valid_idx[:,4]
    sampled_bid = sampled_valid_idx[:,0]
    sampled_valid_idx = torch.stack((sampled_bid,sampled_flat_img_id),-1)
    assert sampled_valid_idx.shape[0] == bs * sample_num
    return sampled_valid_idx

def get_valid_points(data_dict, valid_sample_num=1):
    '''
        If valid_sample_num == -1, use all valid points. Otherwise uniformly sample valid points in a small block.
        valid_idx: (valid_point_num,2), 1st dim is batch idx, 2nd dim is flattened img idx.
    '''
    bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
    if valid_sample_num != -1: # sample valid points
        valid_idx = sample_valid_points(data_dict['valid_mask'], valid_sample_num, block_x=8, block_y=8, data_dict=data_dict)
    else: # get all valid points
        valid_mask_flat = data_dict['valid_mask'].reshape(bs,-1)
        valid_idx = torch.nonzero(valid_mask_flat, as_tuple=False)
    valid_bid = valid_idx[:,0]
    valid_flat_img_id = valid_idx[:,1]
    # get rgb and xyz for valid points.
    valid_xyz = data_dict['xyz'][valid_bid, valid_flat_img_id]
    rgb_img_flat = data_dict['rgb_img'].permute(0,2,3,1).contiguous().reshape(bs,-1,3)
    valid_rgb = rgb_img_flat[valid_bid, valid_flat_img_id]
    #print(len(valid_bid))
    # update intermediate data in data_dict
    data_dict.update({
        'valid_bid': valid_bid,
        'valid_flat_img_id': valid_flat_img_id,
        'valid_xyz': valid_xyz,
        'valid_rgb': valid_rgb, 
    })

def get_occ_vox_bound(data_dict, res=16):
    ##################################
    #  Get occupied voxel in a batch
    ##################################
    # setup grid properties
    xmin = torch.Tensor(constants.XMIN).float().cuda()
    xmax = torch.Tensor(constants.XMAX).float().cuda()
    
    
    min_bb = torch.min(xmax- xmin).item()
    part_size = min_bb / res
    # we need half voxel margin on each side
    xmin = xmin - 0.5 * part_size
    xmax = xmax + 0.5 * part_size
    # get occupied grid
    occ_vox_bid_global_coord, revidx, valid_v_pid, \
    valid_v_rel_coord, idx_grid = batch_get_occupied_idx(
                data_dict['valid_xyz'], data_dict['valid_bid'].unsqueeze(-1),
                xmin=xmin, xmax=xmax, 
                crop_size=part_size, overlap=False)
    # images in current minibatch do not have occupied voxels
    if occ_vox_bid_global_coord.shape[0] == 0:
        print('No occupied voxel', data_dict['item_path'])
        return False
    occ_vox_bid = occ_vox_bid_global_coord[:,0]
    occ_vox_global_coord = occ_vox_bid_global_coord[:,1:]
    ''' compute occupied voxel bound '''
    bound_min = xmin.unsqueeze(0) + occ_vox_global_coord * part_size
    bound_max = bound_min + part_size
    voxel_bound = torch.cat((bound_min,bound_max),1)
    # update data_dict
    data_dict.update({
        'xmin': xmin,
        'part_size': part_size,
        'revidx': revidx,
        'valid_v_pid': valid_v_pid,
        'valid_v_rel_coord': valid_v_rel_coord,
        'occ_vox_bid': occ_vox_bid,
        'occ_vox_global_coord': occ_vox_global_coord,
        'voxel_bound': voxel_bound,    
    })
    return True


def batch_get_occupied_idx(v, batch_id,
    xmin=(0., 0., 0.),
    xmax=(1., 1., 1.),
    crop_size=.125, overlap=False):

    if not torch.is_tensor(xmin):
        xmin = torch.Tensor(xmin).float().to(v.device)
    if not torch.is_tensor(xmax):
        xmax = torch.Tensor(xmax).float().to(v.device)
    # get coords of valid point w.r.t full global grid
    v = v.clone()-xmin.unsqueeze(0)
    # get resolution of voxel grids
    r = torch.ceil((xmax-xmin)/crop_size)
    # if overlap, we need to add r-1 voxel cells in between
    rr = r.long() if not overlap else (2*r-1).long()
    

    # create index grid
    idx_grid = torch.stack(torch.meshgrid(torch.arange(rr[0]),
                                    torch.arange(rr[1]),
                                    torch.arange(rr[2])), dim=-1).to(v.device)

    # shift_idxs for each overlapping grid: shape (1, 1, 3) for non-overlap; (1, 8, 3) for overlap after reshaping 
    shift_idxs = torch.stack(
                    torch.meshgrid(torch.arange(int(overlap)+1),
                    torch.arange(int(overlap)+1),
                    torch.arange(int(overlap)+1)), dim=-1).to(v.device)
    shift_idxs = shift_idxs.reshape(-1,3).unsqueeze(0)

    # get coords of valid point w.r.t each overlapping voxel grid. (np,1 or 8,3)
    v_xyz = v.unsqueeze(1) - shift_idxs * crop_size * 0.5
    v_xmin = v.unsqueeze(1).repeat(1,shift_idxs.shape[1],1)
    # get local voxel coord of voxel of valid point. (np, 1 or 8, 3)
    v_local_coord = torch.floor(v_xyz / crop_size).long()
    # get global voxel coord of voxel of valid point. (np, 1 or 8,3)
    if overlap:
        v_global_coord = 2 * v_local_coord + shift_idxs
        v_voxel_center = v_global_coord * crop_size * 0.5 + 0.5 * crop_size
    else:
        v_global_coord = v_local_coord.clone()
        v_voxel_center = v_global_coord * crop_size + 0.5 * crop_size
    v_rel_coord = v_xmin - v_voxel_center
    # get batch id of voxel of valid point. (np, 1 or 8, 1)
    v_bid = batch_id.clone().unsqueeze(1).repeat(1,shift_idxs.shape[1],1)
    #  we need to build a valid point id tensor so that we can accumulate the features from valid points
    v_pid = torch.arange(v_global_coord.shape[0]).to(v.device)
    v_pid = v_pid.unsqueeze(-1).repeat(1,v_global_coord.shape[1]).unsqueeze(-1).long()
    # check if every voxel of valid point is inside the full global grid.
    valid_mask = torch.ones(v_global_coord.shape[0], v_global_coord.shape[1]).bool().to(v.device)
    for i in range(3):
        valid_mask = torch.logical_and(valid_mask, v_global_coord[:,:, i] >= 0)
        valid_mask = torch.logical_and(valid_mask, v_global_coord[:,:, i] < idx_grid.shape[i])
    # the global voxel coord of valid voxel of valid point, (valid_vox_num, 3)
    valid_v_global_coord = v_global_coord[valid_mask]
    # the valid point index of valid voxel of valid point, (valid_vox_num, 1)
    valid_v_pid = v_pid[valid_mask]
    # the batch id of valid voxel of valid point, (valid_vox_num, 1)
    valid_v_bid = v_bid[valid_mask]
    valid_v_rel_coord = v_rel_coord[valid_mask]
    # concatenate batch id and point grid index before using unique. This step is necessary as we want to make sure
    # same grid index from diff batch id will not be filtered
    valid_v_bid_global_coord = torch.cat((valid_v_bid, valid_v_global_coord), dim=-1)
    # using torch.unique to get occupied voxel coord, and a reverse index. 
    # occ_bid_global_coord[revidx] = valid_v_bid_global_coord
    occ_bid_global_coord, revidx = torch.unique(valid_v_bid_global_coord, dim=0, return_inverse=True)
    return occ_bid_global_coord, revidx, valid_v_pid.reshape(-1), valid_v_rel_coord, idx_grid

def get_miss_ray(data_dict):
    #####################################
    # compute ray dir and img grid index 
    #####################################
    bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
    fx,fy = data_dict['fx'], data_dict['fy']
    cx,cy = data_dict['cx'], data_dict['cy']
    y_ind, x_ind = torch.meshgrid(torch.arange(h), torch.arange(w))
    x_ind = x_ind.unsqueeze(0).repeat(bs,1,1).float().cuda()
    y_ind = y_ind.unsqueeze(0).repeat(bs,1,1).float().cuda()
#    # img grid index, (bs,h*w,2)
    img_ind_flat = torch.stack((x_ind,y_ind),-1).reshape(bs,h*w,2).long()
#    cam_x = x_ind - cx.reshape(-1,1,1)
#    cam_y = (y_ind - cy.reshape(-1,1,1)) * fx.reshape(-1,1,1) / fy.reshape(-1,1,1)
#    cam_z = fx.reshape(-1,1,1).repeat(1,h,w)
#    ray_dir = torch.stack((cam_x,cam_y,cam_z),-1)
#    ray_dir = ray_dir / torch.norm(ray_dir,dim=-1,keepdim=True)
#    ray_dir_flat = ray_dir.reshape(bs,-1,3)
    
    corrupt_uv = data_dict["corrupt_uv"] # b x n_kps x 2
    _, n_kps, _ = corrupt_uv.shape
    corrupt_uv[:, :, 0:1] = corrupt_uv[:, :, 0:1] - cx.reshape(-1, 1, 1)
    corrupt_uv[:, :, 1:2] = (corrupt_uv[:, :, 1:2] - cy.reshape(-1,1,1)) * fx.reshape(-1,1,1) / fy.reshape(-1,1,1)
    cam_z = fx.reshape(-1,1,1).repeat(1,n_kps,1)
    miss_ray_dir = torch.cat([corrupt_uv, cam_z], dim=-1)
    #print("miss_ray_dir.shape", miss_ray_dir.shape)
    miss_ray_dir = miss_ray_dir / torch.norm(miss_ray_dir, dim=-1, keepdim=True)
    miss_ray_dir = miss_ray_dir.view(-1, 3).float()
    ###################################
    # sample miss points 
    # (miss_point_num,2): 1st dim is batch idx, second dim is flatted img idx.
    ###################################
#    corrupt_mask_flat = data_dict['corrupt_mask'].view(bs,-1)
#    miss_idx = torch.nonzero(corrupt_mask_flat, as_tuple=False)
#    print(miss_idx.shape)
    corrupt_uv_int = data_dict["corrupt_uv_int"]
    miss_bid = torch.arange(bs).view(-1, 1).contiguous().repeat(1, n_kps).view(-1).contiguous().cuda()
    miss_flat_img_id = (corrupt_uv_int[:, :, 1] * w + corrupt_uv_int[:, :, 0]).reshape(bs * n_kps).long()
    
#    print("miss_bid", miss_bid.type())
#    print("miss_flat_img_id", miss_flat_img_id.type())
    
    total_miss_sample_num = miss_bid.shape[0]
    
    
    # get ray dir and img index for sampled miss point
    miss_img_ind = img_ind_flat[miss_bid, miss_flat_img_id]
    # update data_dict
    data_dict.update({
        'miss_bid': miss_bid,
        'miss_flat_img_id': miss_flat_img_id,
        'miss_ray_dir': miss_ray_dir,
        'miss_img_ind': miss_img_ind,
        'total_miss_sample_num': total_miss_sample_num 
    })

#def get_miss_ray(data_dict):
#    #####################################
#    # compute ray dir and img grid index 
#    #####################################
#    bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
#    fx,fy = data_dict['fx'], data_dict['fy']
#    cx,cy = data_dict['cx'], data_dict['cy']
#    y_ind, x_ind = torch.meshgrid(torch.arange(h), torch.arange(w))
#    x_ind = x_ind.unsqueeze(0).repeat(bs,1,1).float().cuda()
#    y_ind = y_ind.unsqueeze(0).repeat(bs,1,1).float().cuda()
#    # img grid index, (bs,h*w,2)
#    img_ind_flat = torch.stack((x_ind,y_ind),-1).reshape(bs,h*w,2).long()
#    cam_x = x_ind - cx.reshape(-1,1,1)
#    cam_y = (y_ind - cy.reshape(-1,1,1)) * fx.reshape(-1,1,1) / fy.reshape(-1,1,1)
#    cam_z = fx.reshape(-1,1,1).repeat(1,h,w)
#    ray_dir = torch.stack((cam_x,cam_y,cam_z),-1)
#    ray_dir = ray_dir / torch.norm(ray_dir,dim=-1,keepdim=True)
#    ray_dir_flat = ray_dir.reshape(bs,-1,3)
#    
#    ###################################
#    # sample miss points 
#    # (miss_point_num,2): 1st dim is batch idx, second dim is flatted img idx.
#    ###################################
#    corrupt_mask_flat = data_dict['corrupt_mask'].view(bs,-1)
#    miss_idx = torch.nonzero(corrupt_mask_flat, as_tuple=False)
#    print(miss_idx.shape)
#    
#    total_miss_sample_num = miss_idx.shape[0]
#    
#    miss_bid = miss_idx[:,0]
#    miss_flat_img_id = miss_idx[:,1]
#    print("miss_flat_img_id", miss_flat_img_id.dtype)
#    print("miss_bid", miss_bid.dtype)
#    
#    # get ray dir and img index for sampled miss point
#    miss_ray_dir = ray_dir_flat[miss_bid, miss_flat_img_id]
#    print("miss_ray_dir", miss_ray_dir.shape)
#    
#    miss_img_ind = img_ind_flat[miss_bid, miss_flat_img_id]
#    # update data_dict
#    data_dict.update({
#        'miss_bid': miss_bid,
#        'miss_flat_img_id': miss_flat_img_id,
#        'miss_ray_dir': miss_ray_dir,
#        'miss_img_ind': miss_img_ind,
#        'total_miss_sample_num': total_miss_sample_num 
#    })

def compute_ray_aabb(data_dict):
    ################################## 
    #    Run ray AABB slab test
    #    mask: (occ_vox_num_in_batch, miss_ray_num_in_batch)
    #    dist: (occ_vox_num_in_batch, miss_ray_num_in_batch,2). store in voxel dist and out voxel dist
    ##################################
#    print("missbid",data_dict['miss_bid'].int().type())
#    print("occ vox bid", data_dict['occ_vox_bid'].int().type())
    mask, dist = ray_aabb.forward(data_dict['miss_ray_dir'], data_dict['voxel_bound'], 
                        data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
    mask = mask.long()
    dist = dist.float()

    # get idx of ray-voxel intersect pair
    intersect_idx = torch.nonzero(mask, as_tuple=False)
    occ_vox_intersect_idx = intersect_idx[:,0]
    miss_ray_intersect_idx = intersect_idx[:,1]
    
    #print("unique", len(torch.unique(miss_ray_intersect_idx)))
    #print("all", data_dict["miss_ray_dir"].shape[0])
    
    # images in current mini batch do not have ray occ vox intersection pair.
    if intersect_idx.shape[0] == 0:
        print('No miss ray and occ vox intersection pair', data_dict['item_path'])
        return False
    data_dict.update({
        'mask': mask,
        'dist': dist,
        'occ_vox_intersect_idx': occ_vox_intersect_idx,
        'miss_ray_intersect_idx': miss_ray_intersect_idx,
    })
    return True

def compute_gt(data_dict):
    ###########################################
    #    Compute Groundtruth for position and ray termination label
    ###########################################
    # get gt pos for sampled missing point
    #gt_pos = data_dict['xyz'][data_dict['miss_bid'], data_dict['miss_flat_img_id']] # (bs x n_kps) x 3
    gt_pos = data_dict["corrupt_3d_gt"] # (bs x n_kps) x 3
    
    #print("gt_pos", gt_pos) 
    #print("gt_pos", gt_pos.shape) 
    
    # pcl_mask(i,j) indicates if j-th missing point gt pos inside i-th voxel
    pcl_mask = pcl_aabb.forward(gt_pos.float(), data_dict['voxel_bound'].float(), data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
    pcl_mask = pcl_mask.long()
    # compute gt label for ray termination
    pcl_label = pcl_mask[data_dict['occ_vox_intersect_idx'], data_dict['miss_ray_intersect_idx']]
    pcl_label_float = pcl_label.float()

    # get intersected voxels
    unique_intersect_vox_idx, occ_vox_intersect_idx_nodup2dup = torch.unique(data_dict['occ_vox_intersect_idx'], sorted=True, dim=0, return_inverse=True)
    intersect_voxel_bound = data_dict['voxel_bound'][unique_intersect_vox_idx]
    intersect_vox_bid = data_dict['occ_vox_bid'][unique_intersect_vox_idx]
    # get sampled valid pcl inside intersected voxels
    valid_intersect_mask = pcl_aabb.forward(data_dict['valid_xyz'].float(), intersect_voxel_bound.contiguous().float(), data_dict['valid_bid'].int(), intersect_vox_bid.int().contiguous())
    valid_intersect_mask = valid_intersect_mask.long()
    try:
        valid_intersect_nonzero_idx = torch.nonzero(valid_intersect_mask, as_tuple=False)
    except:
        print(data_dict['valid_xyz'].shape)
        print(valid_intersect_mask.shape)
        print(unique_intersect_vox_idx.shape, intersect_voxel_bound.shape)
        print(data_dict['item_path'])
        
    #print(data_dict['valid_xyz'].shape)
    valid_xyz_in_intersect = data_dict['valid_xyz'][valid_intersect_nonzero_idx[:,1]]
    valid_rgb_in_intersect = data_dict['valid_rgb'][valid_intersect_nonzero_idx[:,1]]
    valid_bid_in_intersect = data_dict['valid_bid'][valid_intersect_nonzero_idx[:,1]]
    # update data_dict
    data_dict.update({
        'gt_pos': gt_pos,
        'pcl_label': pcl_label,
        'pcl_label_float': pcl_label_float,
        'valid_xyz_in_intersect': valid_xyz_in_intersect,
        'valid_rgb_in_intersect': valid_rgb_in_intersect,
        'valid_bid_in_intersect': valid_bid_in_intersect
    })

def prepare_data(batch):
    t0 = time.time()
    data_dict = {}
    rgb_img, xyz_img = batch["rgb_img"].cuda(), batch["xyz_img"].cuda()
    #xyz_img = batch["xyz_img"].cuda()
    t1 = time.time()
    #assert rgb_img.shape == xyz_img.shape
    bs, h, w, _ = xyz_img.shape
    assert h == w
    data_dict.update({"bs" : bs, "h" : h, "w" : w})
    data_dict.update({"fx" : batch["K_depth"][:, 0, 0].view(bs, -1).cuda(), 
                      "fy" : batch["K_depth"][:, 1, 1].view(bs, -1).cuda(),
                      "cx" : batch["K_depth"][:, 0, 2].view(bs, -1).cuda(),
                      "cy" : batch["K_depth"][:, 1, 2].view(bs, -1).cuda(),
                     })
    data_dict.update({
                     "rgb_img" : rgb_img.cuda(),
                     "xyz" : xyz_img.view(bs, -1, 3).cuda()
                     })
    data_dict.update({"path" : batch["rgb_path"]})
    
    
    kps_2d_uv = batch["joints_2D_uv"]
    kps_3d = batch["joints_3D_Z"]
    
    _, n_kps, _ = kps_2d_uv.shape
    kps_2d_uv = kps_2d_uv.cuda() # bs x n_kps x 2
    kps_2d_uv_int = torch.clamp(kps_2d_uv, 0.0, h-1)
    kps_2d_uv_int_flat = kps_2d_uv_int.int().view(-1, 2).contiguous().cuda() # (bs x n_kps) x 2
    kps_2d_uv_int = kps_2d_uv_int.int().contiguous().cuda() 
    #print("kps_2d_uv", kps_2d_uv)
    
    kps_3d = kps_3d.view(-1, 3).contiguous().cuda() # (bs x n_kps) x 3
    
    kps_bid = torch.arange(bs).reshape(-1, 1).repeat(1, n_kps).reshape(-1, 1).cuda() # (bs x nkps) x 1 
    #kps_2d_id = torch.cat([kps_bid, kps_2d_uv[:, 1:2], kps_2d_uv[:, 0:1]], dim=-1)
    
    valid_mask = torch.ones(bs, h, w).float().cuda()
    valid_mask[kps_bid[:, 0].long(), kps_2d_uv_int_flat[:, 1].long(), kps_2d_uv_int_flat[:, 0].long()] = 0.0
    #corrupt_mask = (1.0 - valid_mask.clone()).cuda()
    
    
    valid_mask[torch.where(xyz_img[:, :, :, 2] == 0.0)] = 0.0
    
    
    data_dict.update({
                     "valid_mask" : valid_mask,
                     #"corrupt_mask" : corrupt_mask,
                     "corrupt_3d_gt" : kps_3d, 
                     "corrupt_uv" : kps_2d_uv,
                     "corrupt_uv_int" : kps_2d_uv_int
                     }) 
    
    #print("kps_3d", kps_3d)
    #print("corrupt_uv", kps_2d_uv)
    #print("kps_2d_uv_int", kps_2d_uv_int)
    t2 = time.time()
    return data_dict

#def prepare_data(batch):
#    t0 = time.time()
#    data_dict = {}
#    rgb_img, xyz_img = batch["rgb_img"].cuda(), batch["xyz_img"].cuda()
#    #xyz_img = batch["xyz_img"].cuda()
#    t1 = time.time()
#    #assert rgb_img.shape == xyz_img.shape
#    bs, h, w, _ = xyz_img.shape
#    data_dict.update({"bs" : bs, "h" : h, "w" : w})
#    data_dict.update({"fx" : batch["K_depth"][:, 0, 0].view(bs, -1).cuda(), 
#                      "fy" : batch["K_depth"][:, 1, 1].view(bs, -1).cuda(),
#                      "cx" : batch["K_depth"][:, 0, 2].view(bs, -1).cuda(),
#                      "cy" : batch["K_depth"][:, 1, 2].view(bs, -1).cuda(),
#                     })
#    data_dict.update({
#                     "rgb_img" : rgb_img.cuda(),
#                     "xyz" : xyz_img.view(bs, -1, 3).cuda()
#                     })
#    
#    
#    kps_2d_uv = batch["joints_2D_uv"]
#    kps_3d = batch["joints_3D_Z"]
#    
#    _, n_kps, _ = kps_2d_uv.shape
#    kps_2d_uv = kps_2d_uv.int().view(-1, 2).contiguous().cuda() # (bs x n_kps) x 2
#    kps_3d = kps_3d.view(-1, 3).contiguous().cuda() # (bs x n_kps) x 3
#    
#    kps_bid = torch.arange(bs).reshape(-1, 1).repeat(1, n_kps).reshape(-1, 1).cuda() # (bs x nkps) x 1 
#    #kps_2d_id = torch.cat([kps_bid, kps_2d_uv[:, 1:2], kps_2d_uv[:, 0:1]], dim=-1)
#    
#    valid_mask = torch.ones(bs, h, w).float().cuda()
#    valid_mask[kps_bid[:, 0].long(), kps_2d_uv[:, 1].long(), kps_2d_uv[:, 0].long()] = 0.0
#    corrupt_mask = (1.0 - valid_mask.clone()).cuda()
#    
#    
#    valid_mask[torch.where(xyz_img[:, :, :, 2] == 0.0)] = 0.0
#    
#    
#    data_dict.update({
#                     "valid_mask" : valid_mask,
#                     "corrupt_mask" : corrupt_mask,
#                     "corrupt_3d_gt" : kps_3d, 
#                     }) 
#    t2 = time.time()
#    return data_dict

def get_embedding_ours(data_dict, embed_fn, embeddirs_fn, full_rgb_feat, pnet_model,
                  rgb_embedding_type='ROIAlign', roi_inp_bbox=8, roi_out_bbox=2, pnet_pos_type="rel", pnet_model_type='twostage'):
    ########################### 
    #   Get embedding
    ##########################
    torch.cuda.synchronize()
    t2 = time.time()
    bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
    ''' Positional Encoding '''
    # compute intersect pos
    intersect_dist = data_dict['dist'][data_dict['occ_vox_intersect_idx'], data_dict['miss_ray_intersect_idx']]
    intersect_enter_dist, intersect_leave_dist = intersect_dist[:,0], intersect_dist[:,1]
    intersect_dir = data_dict['miss_ray_dir'][data_dict['miss_ray_intersect_idx']]
    intersect_enter_pos = intersect_dir * intersect_enter_dist.unsqueeze(-1)
    intersect_leave_pos = intersect_dir * intersect_leave_dist.unsqueeze(-1)

    intersect_voxel_bound = data_dict['voxel_bound'][data_dict['occ_vox_intersect_idx']]
    intersect_voxel_center = (intersect_voxel_bound[:,:3] + intersect_voxel_bound[:,3:]) / 2.
    
    # In the original paper, this is abs
    inp_enter_pos = intersect_enter_pos
    inp_leave_pos = intersect_leave_pos
    torch.cuda.synchronize()
    t3 = time.time()
    # positional encoding
    intersect_enter_pos_embed = embed_fn(inp_enter_pos)
    intersect_leave_pos_embed = embed_fn(inp_leave_pos)
    intersect_dir_embed = embeddirs_fn(intersect_dir)    
    
    torch.cuda.synchronize()
    t4 = time.time()
    ''' RGB Embedding ''' 
    miss_ray_intersect_img_ind = data_dict['miss_img_ind'][data_dict['miss_ray_intersect_idx']]
    miss_ray_intersect_bid = data_dict['miss_bid'][data_dict['miss_ray_intersect_idx']]
    #full_rgb_feat = resnet_model(data_dict['rgb_img']) 
    # bs x 32 x H x W (origin_size)
   
    # ROIAlign to pool features
    if rgb_embedding_type == 'ROIAlign':
        # compute input boxes for ROI Align
        miss_ray_intersect_ul = miss_ray_intersect_img_ind - roi_inp_bbox // 2
        miss_ray_intersect_br = miss_ray_intersect_img_ind + roi_inp_bbox // 2
        # clamp is done in original image coords
        miss_ray_intersect_ul[:,0] = torch.clamp(miss_ray_intersect_ul[:,0], min=0., max=w-1)
        miss_ray_intersect_ul[:,1] = torch.clamp(miss_ray_intersect_ul[:,1], min=0., max=h-1)
        miss_ray_intersect_br[:,0] = torch.clamp(miss_ray_intersect_br[:,0], min=0., max=w-1)
        miss_ray_intersect_br[:,1] = torch.clamp(miss_ray_intersect_br[:,1], min=0., max=h-1)
        roi_boxes = torch.cat((miss_ray_intersect_bid.unsqueeze(-1), miss_ray_intersect_ul, miss_ray_intersect_br),-1).float()
        # sampled rgb features for ray-voxel intersect pair. (pair num,rgb_feat_len,roi_out_bbox,roi_out_bbox)
        spatial_scale = 1.0
        intersect_rgb_feat = tv_ops.roi_align(full_rgb_feat, roi_boxes, 
                                output_size=roi_out_bbox,
                                spatial_scale=spatial_scale,
                                aligned=True)
        try:
            intersect_rgb_feat = intersect_rgb_feat.reshape(intersect_rgb_feat.shape[0],-1)
        except:
            print(intersect_rgb_feat.shape)
            print(roi_boxes.shape)
            print(data_dict['miss_ray_intersect_idx'].shape, miss_ray_intersect_bid.shape, miss_ray_intersect_img_ind.shape)
            print(data_dict['total_miss_sample_num'])
            print(data_dict['item_path'])
    else:
        raise NotImplementedError('Does not support RGB embedding type: {}'.format(self.opt.model.rgb_embedding_type))
    torch.cuda.synchronize()
    t5 = time.time()
#    
#    '''  Voxel Embedding '''
    valid_v_rgb = data_dict['valid_rgb'][data_dict['valid_v_pid']]
    if pnet_pos_type == 'rel': # relative position w.r.t voxel center
        pnet_inp = torch.cat((data_dict['valid_v_rel_coord'], valid_v_rgb),-1)
    else:
        raise NotImplementedError('Does not support Pnet pos type: {}'.format(self.opt.model.pnet_pos_type))
    # pointnet forward
    if pnet_model_type == 'twostage':
        occ_voxel_feat = pnet_model(inp_feat=pnet_inp, vox2point_idx=data_dict['revidx'])
        t5_mid = time.time()
    else:
        raise NotImplementedError('Does not support pnet model type: {}'.format(self.opt.model.pnet_model_type))
    intersect_voxel_feat = occ_voxel_feat[data_dict['occ_vox_intersect_idx']]
    torch.cuda.synchronize()
    t6 = time.time()
#    # update data_dict
    data_dict.update({
        'intersect_dir': intersect_dir,
        'intersect_enter_dist': intersect_enter_dist,
        'intersect_leave_dist': intersect_leave_dist,
        'intersect_enter_pos': intersect_enter_pos,
        'intersect_leave_pos': intersect_leave_pos,
        'intersect_enter_pos_embed': intersect_enter_pos_embed,
        'intersect_leave_pos_embed': intersect_leave_pos_embed,
        'intersect_dir_embed': intersect_dir_embed,
        'full_rgb_feat': full_rgb_feat,
        'intersect_rgb_feat': intersect_rgb_feat,
        'intersect_voxel_feat': intersect_voxel_feat
    })

def get_embedding(data_dict, embed_fn, embeddirs_fn, resnet_model, pnet_model,
                  rgb_embedding_type='ROIAlign', roi_inp_bbox=8, roi_out_bbox=2, pnet_pos_type="rel", pnet_model_type='twostage'):
    ########################### 
    #   Get embedding
    ##########################
    torch.cuda.synchronize()
    t2 = time.time()
    bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
    ''' Positional Encoding '''
    # compute intersect pos
    intersect_dist = data_dict['dist'][data_dict['occ_vox_intersect_idx'], data_dict['miss_ray_intersect_idx']]
    intersect_enter_dist, intersect_leave_dist = intersect_dist[:,0], intersect_dist[:,1]

    intersect_dir = data_dict['miss_ray_dir'][data_dict['miss_ray_intersect_idx']]
#    print("intersect_dir.shape", intersect_dir.shape)
#    print("intersect_dir", intersect_dir)
    intersect_enter_pos = intersect_dir * intersect_enter_dist.unsqueeze(-1)
    intersect_leave_pos = intersect_dir * intersect_leave_dist.unsqueeze(-1)
    
#    print("intersect_enter_pos", intersect_enter_pos.shape)
#    print("intersect_leave_pos", intersect_leave_pos.shape)
#    print("intersect_dir.shape", intersect_dir.shape)
#    print("intersect_dist", intersect_dist.shape)
    

    intersect_voxel_bound = data_dict['voxel_bound'][data_dict['occ_vox_intersect_idx']]
    intersect_voxel_center = (intersect_voxel_bound[:,:3] + intersect_voxel_bound[:,3:]) / 2.
    
    # In the original paper, this is abs
    inp_enter_pos = intersect_enter_pos
    inp_leave_pos = intersect_leave_pos
    torch.cuda.synchronize()
    t3 = time.time()
    # positional encoding
    intersect_enter_pos_embed = embed_fn(inp_enter_pos)
    intersect_leave_pos_embed = embed_fn(inp_leave_pos)
    intersect_dir_embed = embeddirs_fn(intersect_dir)    
    
    torch.cuda.synchronize()
    t4 = time.time()
    ''' RGB Embedding ''' 
    miss_ray_intersect_img_ind = data_dict['miss_img_ind'][data_dict['miss_ray_intersect_idx']]
    miss_ray_intersect_bid = data_dict['miss_bid'][data_dict['miss_ray_intersect_idx']]
    full_rgb_feat, global_rgb_feat = resnet_model(data_dict['rgb_img']) 
    
    global_rgb_feat_bid = global_rgb_feat[miss_ray_intersect_bid]
    #print("global_rgb_feat.shape", global_rgb_feat.shape)
    
    # ROIAlign to pool features
    if rgb_embedding_type == 'ROIAlign':
        # compute input boxes for ROI Align
        miss_ray_intersect_ul = miss_ray_intersect_img_ind - roi_inp_bbox // 2
        miss_ray_intersect_br = miss_ray_intersect_img_ind + roi_inp_bbox // 2
        # clamp is done in original image coords
        miss_ray_intersect_ul[:,0] = torch.clamp(miss_ray_intersect_ul[:,0], min=0., max=w-1)
        miss_ray_intersect_ul[:,1] = torch.clamp(miss_ray_intersect_ul[:,1], min=0., max=h-1)
        miss_ray_intersect_br[:,0] = torch.clamp(miss_ray_intersect_br[:,0], min=0., max=w-1)
        miss_ray_intersect_br[:,1] = torch.clamp(miss_ray_intersect_br[:,1], min=0., max=h-1)
        roi_boxes = torch.cat((miss_ray_intersect_bid.unsqueeze(-1), miss_ray_intersect_ul, miss_ray_intersect_br),-1).float()
        # sampled rgb features for ray-voxel intersect pair. (pair num,rgb_feat_len,roi_out_bbox,roi_out_bbox)
        spatial_scale = 1.0
        intersect_rgb_feat = tv_ops.roi_align(full_rgb_feat, roi_boxes, 
                                output_size=roi_out_bbox,
                                spatial_scale=spatial_scale,
                                aligned=True)
        
        try:
            #print("intersect_rgb_feat.shape", intersect_rgb_feat.shape)
            intersect_rgb_feat = intersect_rgb_feat.reshape(intersect_rgb_feat.shape[0],-1)
            intersect_rgb_feat = torch.cat([intersect_rgb_feat, global_rgb_feat_bid],-1)
            
            #print("intersect_rgb_feat.shape", intersect_rgb_feat.shape)
            
        except:
            print(intersect_rgb_feat.shape)
            print(roi_boxes.shape)
            print(data_dict['miss_ray_intersect_idx'].shape, miss_ray_intersect_bid.shape, miss_ray_intersect_img_ind.shape)
            print(data_dict['total_miss_sample_num'])
            print(data_dict['item_path'])
    else:
        raise NotImplementedError('Does not support RGB embedding type: {}'.format(self.opt.model.rgb_embedding_type))
    torch.cuda.synchronize()
    t5 = time.time()
#    
#    '''  Voxel Embedding '''
    valid_v_rgb = data_dict['valid_rgb'][data_dict['valid_v_pid']]
    if pnet_pos_type == 'rel': # relative position w.r.t voxel center
        pnet_inp = torch.cat((data_dict['valid_v_rel_coord'], valid_v_rgb),-1)
    else:
        raise NotImplementedError('Does not support Pnet pos type: {}'.format(self.opt.model.pnet_pos_type))
    # pointnet forward
    if pnet_model_type == 'twostage':
        occ_voxel_feat = pnet_model(inp_feat=pnet_inp, vox2point_idx=data_dict['revidx'])
        t5_mid = time.time()
    else:
        raise NotImplementedError('Does not support pnet model type: {}'.format(self.opt.model.pnet_model_type))
    intersect_voxel_feat = occ_voxel_feat[data_dict['occ_vox_intersect_idx']]
    torch.cuda.synchronize()
    t6 = time.time()
#    # update data_dict
    data_dict.update({
        'intersect_dir': intersect_dir,
        'intersect_enter_dist': intersect_enter_dist,
        'intersect_leave_dist': intersect_leave_dist,
        'intersect_enter_pos': intersect_enter_pos,
        'intersect_leave_pos': intersect_leave_pos,
        'intersect_enter_pos_embed': intersect_enter_pos_embed,
        'intersect_leave_pos_embed': intersect_leave_pos_embed,
        'intersect_dir_embed': intersect_dir_embed,
        'full_rgb_feat': full_rgb_feat,
        "global_rgb_feat" : global_rgb_feat,
        'intersect_rgb_feat': intersect_rgb_feat,
        'intersect_voxel_feat': intersect_voxel_feat
    })
    
#    print("t6 - t5_mid", t6 - t5_mid)
#    print("t6 - t5", t6 - t5)
#    print("t5 - t4", t5 - t4)
#    print("t4 - t3", t4 - t3)
#    print("t3 - t2", t3 - t2)


def get_pred(data_dict, exp_type, epoch, offset_dec,prob_dec,device,offset_range=[0.,1.],maxpool_label_epo=6,scatter_type="Maxpool"):
       ######################################################## 
       # Concat embedding and send to decoder 
       ########################################################
       inp_embed = torch.cat(( data_dict['intersect_voxel_feat'].contiguous(), data_dict['intersect_rgb_feat'].contiguous(),
                               data_dict['intersect_enter_pos_embed'].contiguous(),
                               data_dict['intersect_leave_pos_embed'].contiguous(), data_dict['intersect_dir_embed'].contiguous()),-1)
       pred_offset = offset_dec(inp_embed)
       pred_prob_end = prob_dec(inp_embed)
       
       # scale pred_offset from (0,1) to (offset_range[0], offset_range[1]).
       pred_scaled_offset = pred_offset * (offset_range[1] - offset_range[0]) + offset_range[0]
       pred_scaled_offset = pred_scaled_offset * np.sqrt(3) * data_dict['part_size']
       pair_pred_pos = data_dict['intersect_enter_pos'] + pred_scaled_offset * data_dict['intersect_dir']
       # we detach the pred_prob_end. we don't want pos loss to affect ray terminate score.
       pred_prob_end_softmax = scatter_softmax(pred_prob_end.detach()[:,0], data_dict['miss_ray_intersect_idx'])
       # training uses GT pcl_label to get max_pair_id (voxel with largest prob)
       if exp_type == 'train' and epoch < maxpool_label_epo:
           _, max_pair_id = scatter_max(data_dict['pcl_label_float'], data_dict['miss_ray_intersect_idx'],
                               dim_size=data_dict['total_miss_sample_num'])
       # test/valid uses pred_prob_end_softmax to get max_pair_id (voxel with largest prob)
       else:
           _, max_pair_id = scatter_max(pred_prob_end_softmax, data_dict['miss_ray_intersect_idx'], 
                           dim_size=data_dict['total_miss_sample_num'])
       if scatter_type == 'Maxpool':
           dummy_pos = torch.zeros([1,3]).float().to(device)
           pair_pred_pos_dummy = torch.cat((pair_pred_pos, dummy_pos),0) 
           pred_pos = pair_pred_pos_dummy[max_pair_id]    
           #print("pred_pos", pred_pos.shape)
           
           #print("pred_pos.shape", pred_pos.shape)
       else:
           raise NotImplementedError('Does not support Scatter Type: {}'.format(model.scatter_type))
       
       assert pred_pos.shape[0] == data_dict['total_miss_sample_num']
       # update data_dict
       data_dict.update({
           'pair_pred_pos': pair_pred_pos,
           'max_pair_id': max_pair_id,
           'pred_prob_end': pred_prob_end,
           'pred_prob_end_softmax': pred_prob_end_softmax,
           'pred_pos': pred_pos,
       })
       
def gradient(x):
    # idea from tf.image.image_gradients(image)
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)

    # gradient step=1
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = right - left, bottom - top 
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy

def get_surface_normal(x):
    dx,dy = gradient(x)
    surface_normal = torch.cross(dx, dy, dim=1)
    surface_normal = surface_normal / (torch.norm(surface_normal,dim=1,keepdim=True)+1e-8)
    return surface_normal, dx, dy


def compute_loss(data_dict, exp_type, epoch,device,
                pos_loss_type='single', hard_neg=False, hard_neg_ratio=0.1, pos_loss_fn=nn.L1Loss(),
                prob_loss_type='ray', pos_w=100.0, prob_w=0.5,
                ):
    bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
    ''' position loss '''
    if pos_loss_type == 'single':
        if not hard_neg:
        
            #print("pred_pos", data_dict['pred_pos'].shape)
            #print("gt pos", data_dict['gt_pos'].shape)
            pos_loss = pos_loss_fn(data_dict['pred_pos'], data_dict['gt_pos'])
        else:
            pos_loss_unreduce = torch.mean((data_dict['pred_pos'] - data_dict['gt_pos']).abs(),-1)
            k = int(pos_loss_unreduce.shape[0] * hard_neg_ratio)
            pos_loss_topk,_ = torch.topk(pos_loss_unreduce, k)
            pos_loss = torch.mean(pos_loss_topk)

    ''' Ending probability loss '''
    if prob_loss_type == 'ray':
        pred_prob_end_log_softmax = scatter_log_softmax(data_dict['pred_prob_end'][:,0], data_dict['miss_ray_intersect_idx'])
        pcl_label_idx = torch.nonzero(data_dict['pcl_label'], as_tuple=False).reshape(-1)
        prob_loss_unreduce = -1*pred_prob_end_log_softmax[pcl_label_idx]
        if not hard_neg:
            prob_loss = torch.mean(prob_loss_unreduce)
        else:
            k = int(prob_loss_unreduce.shape[0] * hard_neg_ratio)
            prob_loss_topk,_ = torch.topk(prob_loss_unreduce, k)
            prob_loss = torch.mean(prob_loss_topk)
        
    
    #######################
    # Evaluation Metric
    #######################
    # ending accuracy for missing point
    if exp_type != 'train':
        _, pred_label = scatter_max(data_dict['pred_prob_end_softmax'], data_dict['miss_ray_intersect_idx'],
                                dim_size=data_dict['total_miss_sample_num'])
        _, gt_label = scatter_max(data_dict['pcl_label'], data_dict['miss_ray_intersect_idx'],
                                dim_size=data_dict['total_miss_sample_num'])
        acc = torch.sum(torch.eq(pred_label, gt_label).float()) / torch.numel(pred_label)
    
        # position L2 error: we don't want to consider 0 depth point in the position L2 error.
        zero_mask = torch.sum(data_dict['gt_pos'].abs(),dim=-1)
        zero_mask[zero_mask!=0] = 1.
        elem_num = torch.sum(zero_mask)
        if elem_num.item() == 0:
            err = torch.Tensor([0]).float().to(device)
        else:
            err = torch.sum(torch.sqrt(torch.sum((data_dict['pred_pos'] - data_dict['gt_pos'])**2,-1))*zero_mask) / elem_num
        # compute depth errors following cleargrasp
        zero_mask_idx = torch.nonzero(zero_mask, as_tuple=False).reshape(-1)


    if exp_type != 'train':
        pred_depth = data_dict['pred_pos'][:,2]
        gt_depth = data_dict['gt_pos'][:,2]
        pred = pred_depth[zero_mask_idx]
        gt = gt_depth[zero_mask_idx]
#        else:
#            # scale image to make sure it is same as cleargrasp eval metric
#            gt_xyz = data_dict['xyz_flat'].clone()
#            gt_xyz = gt_xyz.reshape(bs,h,w,3).cpu().numpy()
#            gt_depth = gt_xyz[0,:,:,2]
#            gt_depth = cv2.resize(gt_depth, (256, 144), interpolation=cv2.INTER_NEAREST)
#            gt_depth[np.isnan(gt_depth)] = 0
#            gt_depth[np.isinf(gt_depth)] = 0
#            mask_valid_region = (gt_depth > 0)
#
#            seg_mask = data_dict['corrupt_mask'].cpu().numpy()
#            seg_mask = seg_mask[0].astype(np.uint8)
#            seg_mask = cv2.resize(seg_mask, (256, 144), interpolation=cv2.INTER_NEAREST)
#            mask_valid_region = np.logical_and(mask_valid_region, seg_mask)
#            mask_valid_region = (mask_valid_region.astype(np.uint8) * 255)
#
#            pred_xyz = data_dict['xyz_corrupt_flat'].clone()
#            pred_xyz[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos']
#            pred_xyz = pred_xyz.reshape(bs,h,w,3).cpu().numpy()
#            pred_depth = pred_xyz[0,:,:,2]
#            pred_depth = cv2.resize(pred_depth, (256, 144), interpolation=cv2.INTER_NEAREST)
#
#            gt = torch.from_numpy(gt_depth).float().to(device)
#            pred = torch.from_numpy(pred_depth).float().to(device)
#            mask = torch.from_numpy(mask_valid_region).bool().to(device)
#            gt = gt[mask]
#            pred = pred[mask]

        # compute metrics
        safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))  
        safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
        thresh = torch.max(gt / pred, pred / gt)
        a1 = (thresh < 1.05).float().mean()
        a2 = (thresh < 1.10).float().mean()
        a3 = (thresh < 1.25).float().mean()

        rmse = ((gt - pred)**2).mean().sqrt()
        rmse_log = ((safe_log(gt) - safe_log(pred))**2).mean().sqrt()
        log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean()
        abs_rel = ((gt - pred).abs() / gt).mean()
        mae = (gt - pred).abs().mean()
        sq_rel = ((gt - pred)**2 / gt).mean()

    # loss dict
    loss_dict = {
        'pos_loss': pos_loss,
        'prob_loss': prob_loss,
    }
    if exp_type != 'train':
        loss_dict.update({
            'a1': a1,
            'a2': a2,
            'a3': a3,
            'rmse': rmse,
            'rmse_log': rmse_log,
            'log10': log10,
            'abs_rel': abs_rel,
            'mae': mae,
            'sq_rel': sq_rel,
            'acc': acc,
        })
    return loss_dict

if __name__ == "__main__":
    # we guess the input is batch
    torch.manual_seed(10)
    np.random.seed(10)
    cfg, args = update_config()
    
    testing_dataset = Voxel_dataset(train_dataset_dir="/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/",
                                     val_dataset_dir="/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/",
                                     joint_names=[f"panda_joint_3n_{i+1}" for i in range(14)],
                                     run=[0],
                                     init_mode="train", 
                                     img_type="D",
                                     raw_img_size=tuple([640, 360]),
                                     input_img_size=tuple([384, 216]),
                                     sigma=3.0,
                                     norm_type="min_max",
                                     network_input="XYZ",
                                     network_task="3d_RPE",
                                     depth_range=[500, 3380, 7.5],
                                     depth_range_type="padding",
                                     aug_type="3d",
                                     aug_mode=False)
    
    testing_dataset.train()
    testing_loader = DataLoader(testing_dataset,batch_size=18,
                                  num_workers=1, pin_memory=True, drop_last=True) 
    
    t_list = []       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")               
    
    embed_fn, embeddirs_fn, resnet_model, pnet_model,offset_dec,prob_dec = build_model(device)
    refine_model = build_voxel_refine_network(cfg,device).cuda()
    
    refine_flag = True
    
    
    #embed_fn = embed_fn.cuda()
    #embeddirs_fn = embeddirs_fn.cuda()
    # attention all the rgb here is replaced by xyz
    resnet_model = resnet_model.cuda() 
    #resnet_woff = ResnetSimpleWoff(n_keypoints=14, n_offs=2, full=False).cuda()
    pnet_model = pnet_model.cuda()
    optimizer = torch.optim.AdamW([{'params': resnet_model.parameters(), 'lr': 0.001}, 
                                   {'params': pnet_model.parameters(), 'lr': 0.001}, 
                                   {'params': offset_dec.parameters(), 'lr': 0.001}, 
                                   {'params': prob_dec.parameters(), 'lr': 0.001},
                                   #{'params': resnet_woff.parameters(), 'lr': 0.001}, 
                                   ])
    refine_optimizer = torch.optim.AdamW(refine_model.parameters(), lr=0.001)
    heatmap_criterion = torch.nn.MSELoss()
    woff_criterion = torch.nn.MSELoss()
    #with torch.no_grad():
    all_kps = 0
    occ_kps = 0
    for batch_idx, batch in enumerate(tqdm(testing_loader)): 
#        if batch_idx > 50:
#            break  
        # prepare data
        
        
#        torch.cuda.synchronize()
#        start_time = time.time()
#        if batch_idx >= 1:
#            #print(time.time() - pre_time)
#            print("load time", start_time - pre_time)
#        
#        input_tensor = batch['rgb_img'].to(device).float()
#        joints_3d = batch['joints_3D_Z'].to(device).float()
#        b, c, h, w = input_tensor.size()
#        uv_ind = batch["uv_ind"].to(device).type(torch.int64)
#        uv_off = batch["uv_off"].to(device).float()
#        
#        outputs = resnet_woff(input_tensor)
#        heatmap_pred, off_pred, full_xyz_feat = outputs[0], outputs[1], outputs[2]
#        full_xyz_feat = F.interpolate(full_xyz_feat, (h, w), mode='bilinear', align_corners=False)
#        #print("full_xyz_feat.shape", full_xyz_feat.shape)
#        heatmap_gt = batch['heatmap_25d'].to(device).float()
#        heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
#        heatmap_loss = heatmap_criterion(heatmap_gt, heatmap_pred)
#        
#        off_pred = F.interpolate(off_pred, (h, w), mode='bicubic',align_corners=False) # B x 2 x H x W
#        off_pred = off_pred.permute(0, 2, 3, 1).contiguous()
#        pred_uv_off = ((off_pred[:, :, :, :2]).view(b, -1, 2).contiguous())
#        uv_ind = uv_ind[:, :, :1].expand(uv_ind.size(0), uv_ind.size(1), pred_uv_off.size(2))
#        pred_uv_off = pred_uv_off.gather(1, uv_ind)
#        woff_loss = woff_criterion(pred_uv_off, uv_off) 
        
        
#        torch.cuda.synchronize()
#        t_ = time.time()
        data_dict = prepare_data(batch) # change!
#        torch.cuda.synchronize()
#        end_time = time.time()

        
        # get valid pts other than kps
#        torch.cuda.synchronize()
#        t_ = time.time()
        get_valid_points(data_dict, valid_sample_num=10000) # no edition
#        torch.cuda.synchronize()
#        end_time = time.time()


        
        # get occupied voxel data
#        torch.cuda.synchronize()
#        t_ = time.time()
        get_occ_vox_bound(data_dict, res=12) # no edition
#        torch.cuda.synchronize()
#        end_time = time.time()

        
        # get miss ray data
#        torch.cuda.synchronize()
#        t_ = time.time()
        get_miss_ray(data_dict) # change
#        torch.cuda.synchronize()
#        end_time = time.time()

        
        # ray AABB slab test
#        torch.cuda.synchronize()
#        t_ = time.time()
        intersect_pair_flag = compute_ray_aabb(data_dict)  # change
#        torch.cuda.synchronize()
#        end_time = time.time()
        
        #print("unique", len(torch.unique(miss_ray_intersect_idx)))
    #print("all", data_dict["miss_ray_dir"].shape[0])
        all_kps += data_dict["miss_ray_dir"].shape[0]
        occ_kps += len(torch.unique(data_dict["miss_ray_intersect_idx"]))
#        # compute gt
##        torch.cuda.synchronize()
##        t_ = time.time()
#        compute_gt(data_dict)
##        torch.cuda.synchronize()
##        end_time = time.time()
#
#        
#        # get embedding
##        torch.cuda.synchronize()
##        t_ = time.time()
#        get_embedding(data_dict, embed_fn, embeddirs_fn, resnet_model, pnet_model) 
##        get_embedding_ours(data_dict, embed_fn, embeddirs_fn, full_xyz_feat, pnet_model)
##        torch.cuda.synchronize()
##        end_time = time.time()
#
#        
#        # get pred
#        get_pred(data_dict, "train", batch_idx, offset_dec, prob_dec, device)
#        
#        
##        
#        # calculate loss
#        loss_dict = compute_loss(data_dict, "train", batch_idx,device) 
#        loss = loss_dict["pos_loss"] + loss_dict["prob_loss"]
#        
#
#        if refine_flag:
#            for cur_iter in range(cfg["refine_voxel_network"]["refine_forward_times"]):
#                if cur_iter == 0:
#                    pred_pos_refine = get_pred_refine(data_dict, data_dict['pred_pos'], "train", cur_iter, device, 
#                    refine_model.embeddirs_fn, refine_model.pnet_model, refine_model.embed_fn, refine_model.offset_dec,)
#                else:
#                    pred_pos_refine = get_pred_refine(data_dict, pred_pos_refine, "train", cur_iter, device,
#                    refine_model.embeddirs_fn, refine_model.pnet_model, refine_model.embed_fn, refine_model.offset_dec,)
#        
#            data_dict['pred_pos_refine'] = pred_pos_refine
#            loss_dict_refine = compute_refine_loss(data_dict, "train", batch_idx)
#
#            refine_optimizer.zero_grad()
#            loss_dict_refine["pos_loss"].backward()
#            refine_optimizer.step()
#        else:
#            optimizer.zero_grad()
#            loss.backward()
#            print(loss)
#            optimizer.step()
        
        
        # time accumulated 
#        if batch_idx >= 1:
#            #print(time.time() - pre_time)
#            torch.cuda.synchronize()
#            end_time = time.time()
#            print("network time", end_time - start_time)
#            t_list.append(end_time - start_time)
#        if batch_idx >= 0:
#            torch.cuda.synchronize()
#            pre_time = time.time()
    
        
        
        pass 
    print(np.mean(t_list)) 
    print(all_kps)
    print(occ_kps)
    
    
    # considering what is involved in a batch
#    depth_path = f"/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/00500/0000_simDepthImage.exr"
#    png_path = f"/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/00500/0000_color.png"
#    json_file = f"/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/00500/0000_meta.json"
#    
#    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32) # H x W 
#    color_img = cv2.imread(png_path)[None, :, :, :] # bs x H x W x 3
#    bs, H, W, _ = color_img.shape
#    xyz_img = DepthToPointCloud(depth_img[:, :, None], f=502.30).reshape(bs, H * W, 3) # (bs x (HW) x 3)
#    
#    with open(json_file, 'r') as fd:
#        keypoints_json = json.load(fd)[0]["joints_3n_fixed_42"]
#    keypoints_data_3d = [keypoints_json[idx]["location_wrt_cam"] for idx in range(len(keypoints_json))]
#    keypoints_data_3d = torch.from_numpy(np.array(keypoints_data_3d))
#    camera_intrinsics = torch.from_numpy(np.array([[502.30, 0.0, 319.5], [0.0, 502.30, 179.5], [0.0, 0.0, 1.0]]))
#    keypoints_data_2d = ((camera_intrinsics @ keypoints_data_3d.T).T / keypoints_data_3d[:, 2:3])[:, :2].int()
#    
#    
#    keypoints_data_2d_uv = keypoints_data_2d.view(-1, 2).contiguous() # (bs x n_kps) x 2
#    keypoints_data_3d = keypoints_data_3d.view(-1, 3).contiguous() # (bs x n_kps) x 3
#    
#    n_kps, _ = keypoints_data_2d.shape
#    keypoints_data_bid = torch.arange(bs).reshape(-1, 1).repeat(1, n_kps).reshape(-1, 1) # (bs x nkps) x 1 
#    #print(keypoints_data_2d_uv)
#    keypoints_data_2d_id = torch.cat([keypoints_data_bid, keypoints_data_2d_uv[:, 1:2], keypoints_data_2d_uv[:, 0:1]], dim=-1)
#    #print(keypoints_data_2d_id)
#    
#    valid_mask = torch.ones(bs, H, W).float()
#    valid_mask[keypoints_data_2d_id[:, 0], keypoints_data_2d_id[:, 1], keypoints_data_2d_id[:, 2]] = 0.0
#    corrupt_mask = 1 - valid_mask
#    #print(torch.where(corrupt_mask > 0.0))
#    
#    for i in range(1000):
#        t1 = time.time()
#        data_dict = {
#                    "bs" : bs, "h" : H, "w" : W, 
#                    "fx" : torch.tensor([502.30]).view(bs, -1).cuda(), 
#                    "fy" : torch.tensor([502.30]).view(bs, -1).cuda(),  
#                    "cx" : torch.tensor([319.50]).view(bs, -1).cuda(), 
#                    "cy" : torch.tensor([179.50]).view(bs, -1).cuda(),
#                    "rgb_img" : torch.from_numpy(color_img).cuda(), # bs x H x W x 3
#                    "xyz" : torch.from_numpy(xyz_img).cuda(), # bs x (H x W) x 3
#                    "valid_mask" : valid_mask.cuda(), # bs x H x W  for non keypoints
#                    "corrupt_mask" : (1 - valid_mask).cuda(), # for keypoints
#                    "corrupt_3d_gt" : keypoints_data_3d.cuda(),
#                    }
#        
#        # for testing
#          
#        # get valid pts other than kps
#        get_valid_points(data_dict, valid_sample_num=-1)
#        
#        # get occupied voxel data
#        get_occ_vox_bound(data_dict, res=16)
#        
#        # get miss ray data
#        get_miss_ray(data_dict)
#        
#        # ray AABB slab test
#        intersect_pair_flag = compute_ray_aabb(data_dict) 
#        
#        # compute gt
#        compute_gt(data_dict)
#
#        print(time.time() - t1)
#        pass 
    



