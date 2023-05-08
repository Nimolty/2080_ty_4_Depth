import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from extensions.ray_aabb.jit import ray_aabb
from extensions.pcl_aabb.jit import pcl_aabb

import torch
import torch.nn.functional as F

import numpy as np
import time
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import constants

def DepthToPointCloud(depthImage, f):
    H, W, _ = depthImage.shape
    depthImage_copy = deepcopy(depthImage[:, :, 0])
    du = W//2 - 0.5
    dv = H//2 - 0.5

    pointCloud = np.zeros((H, W, 3)) # 存x,y,z
    IndexX = np.arange(0, W)[None, :] - du
    IndexY = np.arange(0, H)[:, None] - dv
    
    pointCloud[:, :, 0] = depthImage_copy * IndexX / f
    pointCloud[:, :, 1] = depthImage_copy * IndexY / f
    pointCloud[:, :, 2] = depthImage_copy
    
    return pointCloud

def sample_valid_points(valid_mask, sample_num, block_x=8, block_y=8):
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
    for i in range(bid_interval.shape[0]-1):
        sid = bid_interval[i]
        eid = bid_interval[i+1]
        cur_cnt = eid - sid
        if cur_cnt < sample_num:
            mult = np.ceil(float(sample_num)/float(cur_cnt)) - 1
            cur_points_idx = torch.arange(sid,eid).long().to(valid_mask.device)
            rand_pool = cur_points_idx.repeat(int(mult))
            nextra = sample_num - cur_cnt
            rand_pool_idx = np.random.choice(rand_pool.shape[0], nextra, replace=False)
            extra_idx = rand_pool[rand_pool_idx]
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
        valid_idx = sample_valid_points(data_dict['valid_mask'], valid_sample_num, block_x=8, block_y=8)
    else: # get all valid points
        valid_mask_flat = data_dict['valid_mask'].reshape(bs,-1)
        valid_idx = torch.nonzero(valid_mask_flat, as_tuple=False)
    valid_bid = valid_idx[:,0]
    valid_flat_img_id = valid_idx[:,1]
    # get rgb and xyz for valid points.
    valid_xyz = data_dict['xyz'][valid_bid, valid_flat_img_id]
    rgb_img_flat = data_dict['rgb_img'].permute(0,2,3,1).contiguous().reshape(bs,-1,3)
    valid_rgb = rgb_img_flat[valid_bid, valid_flat_img_id]
    # update intermediate data in data_dict
    data_dict.update({
        'valid_bid': valid_bid,
        'valid_flat_img_id': valid_flat_img_id,
        'valid_xyz': valid_xyz,
        'valid_rgb': valid_rgb, 
    })

def get_occ_vox_bound(data_dict):
    ##################################
    #  Get occupied voxel in a batch
    ##################################
    # setup grid properties
    xmin = torch.Tensor(constants.XMIN).float().cuda()
    xmax = torch.Tensor(constants.XMAX).float().cuda()
    
#    print(constants.XMIN)
#    print(constants.XMAX)
    
    min_bb = torch.min(xmax- xmin).item()
    part_size = min_bb / 16
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
    
    #print(crop_size)
    print(rr) 

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
    # img grid index, (bs,h*w,2)
    img_ind_flat = torch.stack((x_ind,y_ind),-1).reshape(bs,h*w,2).long()
    cam_x = x_ind - cx.reshape(-1,1,1)
    cam_y = (y_ind - cy.reshape(-1,1,1)) * fx.reshape(-1,1,1) / fy.reshape(-1,1,1)
    cam_z = fx.reshape(-1,1,1).repeat(1,h,w)
    ray_dir = torch.stack((cam_x,cam_y,cam_z),-1)
    ray_dir = ray_dir / torch.norm(ray_dir,dim=-1,keepdim=True)
    ray_dir_flat = ray_dir.reshape(bs,-1,3)
    
    ###################################
    # sample miss points 
    # (miss_point_num,2): 1st dim is batch idx, second dim is flatted img idx.
    ###################################
    corrupt_mask_flat = data_dict['corrupt_mask'].view(bs,-1)
    miss_idx = torch.nonzero(corrupt_mask_flat, as_tuple=False)
    
    total_miss_sample_num = miss_idx.shape[0]
    miss_bid = miss_idx[:,0]
    miss_flat_img_id = miss_idx[:,1]
    # get ray dir and img index for sampled miss point
    miss_ray_dir = ray_dir_flat[miss_bid, miss_flat_img_id]
    miss_img_ind = img_ind_flat[miss_bid, miss_flat_img_id]
    # update data_dict
    data_dict.update({
        'miss_bid': miss_bid,
        'miss_flat_img_id': miss_flat_img_id,
        'miss_ray_dir': miss_ray_dir,
        'miss_img_ind': miss_img_ind,
        'total_miss_sample_num': total_miss_sample_num 
    })

def compute_ray_aabb(data_dict):
    ################################## 
    #    Run ray AABB slab test
    #    mask: (occ_vox_num_in_batch, miss_ray_num_in_batch)
    #    dist: (occ_vox_num_in_batch, miss_ray_num_in_batch,2). store in voxel dist and out voxel dist
    ##################################
    mask, dist = ray_aabb.forward(data_dict['miss_ray_dir'], data_dict['voxel_bound'], 
                        data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
    mask = mask.long()
    dist = dist.float()

    # get idx of ray-voxel intersect pair
    intersect_idx = torch.nonzero(mask, as_tuple=False)
    occ_vox_intersect_idx = intersect_idx[:,0]
    miss_ray_intersect_idx = intersect_idx[:,1]
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
    gt_pos = data_dict['xyz'][data_dict['miss_bid'], data_dict['miss_flat_img_id']]
    
    print("gt_pos", gt_pos.shape)
    
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

def get_embedding(self, data_dict):
    ########################### 
    #   Get embedding
    ##########################
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
    if self.opt.model.intersect_pos_type == 'rel':
        inp_enter_pos = intersect_enter_pos - intersect_voxel_center
        inp_leave_pos = intersect_leave_pos - intersect_voxel_center
    else:
        inp_enter_pos = intersect_enter_pos
        inp_leave_pos = intersect_leave_pos

    # positional encoding
    intersect_enter_pos_embed = self.embed_fn(inp_enter_pos)
    intersect_leave_pos_embed = self.embed_fn(inp_leave_pos)
    intersect_dir_embed = self.embeddirs_fn(intersect_dir)    
    
    ''' RGB Embedding ''' 
    miss_ray_intersect_img_ind = data_dict['miss_img_ind'][data_dict['miss_ray_intersect_idx']]
    miss_ray_intersect_bid = data_dict['miss_bid'][data_dict['miss_ray_intersect_idx']]
    full_rgb_feat = self.resnet_model(data_dict['rgb_img'])
    # ROIAlign to pool features
    if self.opt.model.rgb_embedding_type == 'ROIAlign':
        # compute input boxes for ROI Align
        miss_ray_intersect_ul = miss_ray_intersect_img_ind - self.opt.model.roi_inp_bbox // 2
        miss_ray_intersect_br = miss_ray_intersect_img_ind + self.opt.model.roi_inp_bbox // 2
        # clamp is done in original image coords
        miss_ray_intersect_ul[:,0] = torch.clamp(miss_ray_intersect_ul[:,0], min=0., max=w-1)
        miss_ray_intersect_ul[:,1] = torch.clamp(miss_ray_intersect_ul[:,1], min=0., max=h-1)
        miss_ray_intersect_br[:,0] = torch.clamp(miss_ray_intersect_br[:,0], min=0., max=w-1)
        miss_ray_intersect_br[:,1] = torch.clamp(miss_ray_intersect_br[:,1], min=0., max=h-1)
        roi_boxes = torch.cat((miss_ray_intersect_bid.unsqueeze(-1), miss_ray_intersect_ul, miss_ray_intersect_br),-1).float()
        # sampled rgb features for ray-voxel intersect pair. (pair num,rgb_feat_len,roi_out_bbox,roi_out_bbox)
        spatial_scale = 1.0
        intersect_rgb_feat = tv_ops.roi_align(full_rgb_feat, roi_boxes, 
                                output_size=self.opt.model.roi_out_bbox,
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

    '''  Voxel Embedding '''
    valid_v_rgb = data_dict['valid_rgb'][data_dict['valid_v_pid']]
    if self.opt.model.pnet_pos_type == 'rel': # relative position w.r.t voxel center
        pnet_inp = torch.cat((data_dict['valid_v_rel_coord'], valid_v_rgb),-1)
    else:
        raise NotImplementedError('Does not support Pnet pos type: {}'.format(self.opt.model.pnet_pos_type))
    # pointnet forward
    if self.opt.model.pnet_model_type == 'twostage':
        occ_voxel_feat = self.pnet_model(inp_feat=pnet_inp, vox2point_idx=data_dict['revidx'])
    else:
        raise NotImplementedError('Does not support pnet model type: {}'.format(self.opt.model.pnet_model_type))
    intersect_voxel_feat = occ_voxel_feat[data_dict['occ_vox_intersect_idx']]

    # update data_dict
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



if __name__ == "__main__":
    torch.manual_seed(10)
    
    depth_path = f"/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/00500/0000_simDepthImage.exr"
    png_path = f"/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/00500/0000_color.png"
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32) # H x W 
    color_img = cv2.imread(png_path)[None, :, :, :] # 1 x H x W x 3
    
    bs, H, W, _ = color_img.shape
    xyz_img = DepthToPointCloud(depth_img[:, :, None], f=502.30).reshape(bs, H * W, 3)
    valid_mask = torch.zeros(bs, H, W).cuda()
    valid_mask[0][np.where(depth_img > 1e-3)] = 1.0

    t = []
    for i in range(1000):
        data_dict = {"bs" : valid_mask.shape[0], "h" : H, "w" : W, 
                 "fx" : torch.tensor([502.30]).cuda(), 
                 "fy" : torch.tensor([502.30]).cuda(),  
                 "cx" : torch.tensor([319.50]).cuda(), 
                 "cy" : torch.tensor([179.50]).cuda(),
                 "rgb_img" : torch.from_numpy(color_img).cuda(), # bs x H x W x 3
                 "xyz" : torch.from_numpy(xyz_img).cuda(), # bs x (H x W) x 3
                 "valid_mask" : valid_mask, # bs x H x W 
                 "corrupt_mask" : 1 - valid_mask,
                }
        t1 = time.time()
        # get valid points data
        get_valid_points(data_dict, valid_sample_num=-1)
        # get occupied voxel data
        get_occ_vox_bound(data_dict)
        # get miss ray data
        get_miss_ray(data_dict)
        # ray AABB slab test
        intersect_pair_flag = compute_ray_aabb(data_dict) 
        
        t.append(time.time() - t1) 
        #print(time.time() - t1)
        
        # compute gt
        compute_gt(data_dict)
        
#        print(data_dict['occ_vox_intersect_idx'].shape)
#        print(data_dict['miss_ray_intersect_idx'].shape)
#        
        #print(time.time() - t1)  
    print(np.mean(t))
    #valid_ids = sample_valid_points(valid_mask, sample_num=10000, block_x=8, block_y=8)
    #print(valid_ids.shape)

    