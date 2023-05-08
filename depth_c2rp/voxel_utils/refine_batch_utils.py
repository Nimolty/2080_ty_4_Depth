import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from depth_c2rp.voxel_utils.extensions.ray_aabb.jit import ray_aabb
from depth_c2rp.voxel_utils.extensions.pcl_aabb.jit import pcl_aabb

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, Dataset)
import torchvision.ops as tv_ops
import torch.nn as nn 
from torch_scatter import scatter, scatter_softmax, scatter_max, scatter_log_softmax

import numpy as np
import time
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import depth_c2rp.voxel_utils.constants as constants
import json
from tqdm import tqdm

def get_pred_refine(data_dict, pred_pos, exp_type, cur_iter, device, 
                    refine_embeddirs_fn, refine_pnet_model, refine_embed_fn, refine_offset_dec,
                    refine_perturb=True, refine_perturb_prob=0.8, refine_rgb_embedding_type='ROIAlign',
                    refine_roi_inp_bbox=8, roi_out_bbox=2, refine_pnet_pos_type='rel', mask_type="all", refine_use_all_pix=True,
                    refine_pnet_model_type='twostage', refine_offset_range=[-0.2, 0.2],
                    ):
    bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
    concat_dummy = lambda feat: torch.cat((feat, torch.zeros([1,feat.shape[1]]).to(feat.dtype).to(device)),0)
    # manually perturb prediction by adding noise, we only perturb in 1st iter.
    if exp_type == 'train' and refine_perturb and cur_iter == 0 and np.random.random() < refine_perturb_prob:
        prob = np.random.random()
        if prob < 0.5:
            noise = np.random.random() * (0 + 0.05) - 0.05
        elif prob < 0.8:
            noise = np.random.random() * (0.05 - 0)
        elif prob < 0.9:
            noise = np.random.random() * (-0.05 + 0.1) - 0.1
        else:
            noise = np.random.random() * (0.1 - 0.05) + 0.05

        pred_pos = pred_pos + noise * data_dict['miss_ray_dir']
    # recompute voxel ending id
    pred_occ_mask = pcl_aabb.forward(pred_pos, data_dict['voxel_bound'], data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
    pred_occ_mask = pred_occ_mask.long()
    pred_occ_mask_idx = torch.nonzero(pred_occ_mask, as_tuple=False)
    occ_vox_intersect_idx_dummy = concat_dummy(data_dict['occ_vox_intersect_idx'].unsqueeze(-1))
    end_voxel_id = occ_vox_intersect_idx_dummy[data_dict['max_pair_id']].reshape(-1)
    scatter(pred_occ_mask_idx[:,0], pred_occ_mask_idx[:,1], out=end_voxel_id, reduce='max')

    # dir embed
    intersect_dir_embed_end = refine_embeddirs_fn(data_dict['miss_ray_dir'])    
    # rgb embed 
    miss_ray_img_ind = data_dict['miss_img_ind']
    miss_ray_bid = data_dict['miss_bid']
    
    global_feat = data_dict["global_rgb_feat"]
    # ROIAlign to pool features
    if refine_rgb_embedding_type == 'ROIAlign':
        # compute input boxes for ROI Align
        miss_ray_ul = miss_ray_img_ind - refine_roi_inp_bbox // 2
        miss_ray_br = miss_ray_img_ind + refine_roi_inp_bbox // 2
        # clamp is done in original image coords
        miss_ray_ul[:,0] = torch.clamp(miss_ray_ul[:,0], min=0., max=w-1)
        miss_ray_ul[:,1] = torch.clamp(miss_ray_ul[:,1], min=0., max=h-1)
        miss_ray_br[:,0] = torch.clamp(miss_ray_br[:,0], min=0., max=w-1)
        miss_ray_br[:,1] = torch.clamp(miss_ray_br[:,1], min=0., max=h-1)
        roi_boxes = torch.cat((miss_ray_bid.unsqueeze(-1), miss_ray_ul, miss_ray_br),-1).float()
        # sampled rgb features for ray-voxel intersect pair. (pair num,rgb_feat_len,roi_out_bbox,roi_out_bbox)
        spatial_scale = 1.0
        intersect_rgb_feat_end = tv_ops.roi_align(data_dict['full_rgb_feat'], roi_boxes, 
                                output_size=roi_out_bbox,
                                spatial_scale=spatial_scale,
                                aligned=True)
        try:
            intersect_rgb_feat_end = intersect_rgb_feat_end.reshape(intersect_rgb_feat_end.shape[0],-1)
            intersect_rgb_feat_end = torch.cat([intersect_rgb_feat_end, global_feat[miss_ray_bid]], dim=-1)
            #print("intersect_rgb_feat_end", intersect_rgb_feat_end.shape)
        except:
            print(data_dict['item_path'])
    else:
        raise NotImplementedError('Does not support RGB embedding type: {}'.format(self.opt.model.rgb_embedding_type))
    # miss point rgb
    rgb_img_flat = data_dict['rgb_img'].permute(0,2,3,1).contiguous().reshape(data_dict['bs'],-1,3)
    miss_rgb = rgb_img_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]
    # get miss point predicted ending voxel center
    occ_voxel_bound = data_dict['voxel_bound']
    end_voxel_bound = occ_voxel_bound[end_voxel_id]
    end_voxel_center = (end_voxel_bound[:, :3] + end_voxel_bound[:, 3:]) / 2.
    # prep_inp
    if refine_pnet_pos_type == 'rel':
        pred_rel_xyz = pred_pos - end_voxel_center
        pred_inp = torch.cat((pred_rel_xyz, miss_rgb),1)
    else:
        pred_inp = torch.cat((pred_pos, miss_rgb),1)
    if exp_type != 'train' and mask_type == 'all' and refine_use_all_pix == False:
        inp_zero_mask = 1 - data_dict['valid_mask']
        zero_pixel_idx = torch.nonzero(inp_zero_mask, as_tuple=False)
        pred_inp_img = pred_inp.reshape(bs,h,w,pred_inp.shape[-1])
        new_pred_inp = pred_inp_img[zero_pixel_idx[:,0],zero_pixel_idx[:,1],zero_pixel_idx[:,2]]
        end_voxel_id_img = end_voxel_id.reshape(bs,h,w)
        new_end_voxel_id = end_voxel_id_img[zero_pixel_idx[:,0],zero_pixel_idx[:,1],zero_pixel_idx[:,2]]
    else:
        new_pred_inp = pred_inp
        new_end_voxel_id = end_voxel_id
    

    # pnet inp
    valid_v_rgb = data_dict['valid_rgb'][data_dict['valid_v_pid']]
    if refine_pnet_pos_type == 'rel': # relative position w.r.t voxel center
        pnet_inp = torch.cat((data_dict['valid_v_rel_coord'], valid_v_rgb),-1)
    elif refine_pnet_pos_type == 'abs': # absolute position
        valid_v_xyz = data_dict['valid_xyz'][data_dict['valid_v_pid']]
        pnet_inp = torch.cat((valid_v_xyz, valid_v_rgb),-1)
    else:
        raise NotImplementedError('Does not support Pnet pos type: {}'.format(refine_pnet_pos_type))
    # concat pnet_inp and pred_inp, and update revidx
    final_pnet_inp = torch.cat((pnet_inp, new_pred_inp),0)
    final_revidx = torch.cat((data_dict['revidx'],new_end_voxel_id),0)
    # pointnet forward
    if refine_pnet_model_type == 'twostage':
        occ_voxel_feat = refine_pnet_model(inp_feat=final_pnet_inp, vox2point_idx=final_revidx)
    else:
        raise NotImplementedError('Does not support Pnet model type: {}'.format(refine_pnet_model_type))
    intersect_voxel_feat_end = occ_voxel_feat[end_voxel_id]

    # pos embed
    # all abs
    enter_pos = pred_pos
    intersect_pos_embed_end = refine_embed_fn(enter_pos)
    # concat inp
    inp_embed = torch.cat((intersect_voxel_feat_end, intersect_rgb_feat_end, 
                    intersect_pos_embed_end, intersect_dir_embed_end),-1)
    
    print("inp-embed", inp_embed.shape)
    
    pred_refine_offset = refine_offset_dec(inp_embed)
    pred_scaled_refine_offset = pred_refine_offset * (refine_offset_range[1] - refine_offset_range[0]) + refine_offset_range[0]
    pred_pos_refine = pred_pos + pred_scaled_refine_offset * data_dict['miss_ray_dir']
    return pred_pos_refine

        
def compute_refine_loss(data_dict, exp_type, epoch,
                        refine_pos_loss_type='single', refine_hard_neg=False, refine_hard_neg_ratio=0.0, refine_pos_loss_fn = nn.L1Loss(),
                        
                        
                        ):
    bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
    ''' position loss '''
    if refine_pos_loss_type == 'single':
        if not refine_hard_neg:
            pos_loss = refine_pos_loss_fn(data_dict['pred_pos_refine'], data_dict['gt_pos'])
        else:
            pos_loss_unreduce = torch.mean((data_dict['pred_pos_refine'] - data_dict['gt_pos']).abs(),-1)
            k = int(pos_loss_unreduce.shape[0] * refine_hard_neg_ratio)
            pos_loss_topk,_ = torch.topk(pos_loss_unreduce, k)
            pos_loss = torch.mean(pos_loss_topk)
    else:
        raise NotImplementedError('Does not support pos_loss_type for refine model'.format(self.opt.loss.pos_loss_type))
    
    #######################
    # Evaluation Metric
    #######################
    # position L2 error: we don't want to consider 0 depth point in the position L2 error.
    if exp_type != 'train':
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

    # update data_dict
    # loss dict
    loss_dict = {
        'pos_loss': pos_loss,
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
        })

    return loss_dict



