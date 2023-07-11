import numpy as np
import os
import torch
import cv2
from depth_c2rp.utils.utils import quaternionToRotation
from depth_c2rp.utils.utils import batch_quaternion_matrix, compute_concat_loss
from depth_c2rp.utils.spdh_network_utils import compute_rigid_transform
import math

def batch_outlier_removal_pose(joints_2D, joints_3d_rob_pred, joints_3d_pred_norm, joints_3d_pred, size):
    # expect all tensors
    # joints_2D : B x N x 2
    # joints_3d_pred : B X N X 3
    # joints_3d_norm : B X N X 3
    H, W = size
    assert H == W
    B, N, _ = joints_2D.shape
    pose_list = []
    
    for b in range(B):
        index = torch.where(torch.abs((joints_2D[b] - H // 2) > H // 2))
        index = torch.unique(index[0])
        joints_2D[b][index] = -999.999
        
        valid_index = torch.unique(torch.where(joints_2D[b] > -999.999)[0])
        pose_pred = (compute_rigid_transform(joints_3d_rob_pred[b][valid_index],joints_3d_pred_norm[b][valid_index])).unsqueeze(0)
        pose_pred[:, :3, 3] = joints_3d_pred[b][0].reshape(1,3) + pose_pred[:, :3, 3]
        pose_list.append(pose_pred)
    return torch.cat(pose_list, dim=0) 

def add_from_pose(quaternion, trans, x3d_wrt_cam, x3d_wrt_rob):
    # x3d_wrt_cam : (N, 3) ; x3d_wrt_rob : (N, 3)
    rot = np.array(quaternionToRotation(quaternion))
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, -1] = trans
    kp_pos_gt_homog = np.hstack(
        (
            x3d_wrt_rob,
            np.ones((x3d_wrt_rob.shape[0], 1)),
        )
    )
    kp_pos_aligned = np.transpose(np.matmul(transform, np.transpose(kp_pos_gt_homog)))[
        :, :3
    ]
    kp_3d_errors = kp_pos_aligned - x3d_wrt_cam
    kp_3d_l2_errors = np.linalg.norm(kp_3d_errors, axis=1)
    add = np.mean(kp_3d_l2_errors)
    return add
    
def batch_add_from_pose(dt_joints_wrt_cam, gt_joints_wrt_cam):
    # dt_joints_wrt_cam, gt_joints_wrt_cam.shape : B x N x 3
    kp_3d_errors = dt_joints_wrt_cam  - gt_joints_wrt_cam
    kp_3d_l2_errors = np.linalg.norm(kp_3d_errors, axis=-1) # B x N
    kp_3d_lr_errors_mean = np.mean(kp_3d_l2_errors, axis=1) # B
    return kp_3d_lr_errors_mean.tolist()

def batch_repeat_add_from_pose(dt_joints_wrt_cam, gt_joints_wrt_cam, bs, num_samples):
    # dt_joints_wrt_cam, gt_joints_wrt_cam.shape : (bsxnum_samples) x N x 3
    assert 3 == dt_joints_wrt_cam.shape[2]
    kp_3d_errors = dt_joints_wrt_cam  - gt_joints_wrt_cam
    kp_3d_errors = kp_3d_errors.reshape(num_samples, bs, -1, 3) # num_samples x bs x N x 3
    kp_3d_errors = kp_3d_errors.transpose(1, 2, 0, 3) # bs x N x num_samples x 3
    kp_3d_l2_errors = np.linalg.norm(kp_3d_errors, axis=-1) # bs x N x num_samples
    kp_3d_lr_errors_mean = np.mean(kp_3d_l2_errors, axis=-1) # bs x N
    kp_3d_lr_errors_mean = np.min(kp_3d_lr_errors_mean, axis=-1) # bs
    return kp_3d_lr_errors_mean.tolist()

def flat_add_from_pose(dt_joints_wrt_cam, gt_joints_wrt_cam):
    # dt_joints_wrt_cam, gt_joints_wrt_cam : N x 3
    kp_3d_errors = dt_joints_wrt_cam - gt_joints_wrt_cam
    kp_3d_l2_errors = np.linalg.norm(kp_3d_errors, axis=-1)
    return kp_3d_l2_errors.tolist()

def batch_repeat_mAP_from_pose(dt_joints_wrt_cam, gt_joints_wrt_cam, bs, num_samples, thresholds):
    this_mAP = []
    assert 3 == dt_joints_wrt_cam.shape[2]
    kp_3d_errors = dt_joints_wrt_cam  - gt_joints_wrt_cam
    kp_3d_errors = kp_3d_errors.reshape(num_samples, bs, -1, 3) # num_samples x bs x N x 3
    kp_3d_errors = kp_3d_errors.transpose(1, 2, 0, 3) # bs x N x num_samples x 3
    kp_3d_l2_errors = np.linalg.norm(kp_3d_errors, axis=-1) # bs x N x num_samples
    kp_3d_lr_errors_mean = np.mean(kp_3d_l2_errors, axis=-1) # bs x N
    dist_3D = np.min(kp_3d_lr_errors_mean, axis=-1) # bs
    for thresh in thresholds:
        avg_AP = (dist_3D < thresh)
        this_mAP.append(avg_AP.tolist())
    return np.array(this_mAP).T.tolist()

def batch_mAP_from_pose(dt_joints_wrt_cam, gt_joints_wrt_cam, thresholds):
    # dt_joints_wrt_cam, gt_joints_wrt_cam.shape : B x N x 3
    this_mAP = []
    dist_3D = np.sqrt(np.sum((dt_joints_wrt_cam - gt_joints_wrt_cam)**2, axis=-1)) # B x N 
    dist_3D = np.mean(dist_3D, axis=-1) # B
    for thresh in thresholds:
        avg_AP = (dist_3D < thresh)
        this_mAP.append(avg_AP.tolist())
    #print(np.array(this_mAP).T.shape)
    return np.array(this_mAP).T.tolist()

def batch_repeat_acc_from_joint_angles(dt_joints_pos, gt_joints_pos, bs, num_samples, thresholds):
    # dt_joints_pos, gt_joints_pos.shape : (bsxnum_samples) x 7 x 1
    this_acc = []
    dist_angles = np.abs(dt_joints_pos - gt_joints_pos)
    dist_angles = dist_angles.reshape(num_samples, bs, -1, 1) # num_samples x bs x 7 x 1
    dist_angles = dist_angles.transpose(1, 2, 0, 3)[:, :, :, 0] # bs x 7 x num_samples
    dist_angles = np.min(dist_angles, axis=-1) # bs x 7
    #print("dist_angles.shape", dist_angles.shape)
    for thresh in thresholds:
        radian_thresh = math.radians(thresh)
        avg_acc = np.mean(dist_angles < radian_thresh, axis=-1) 
        this_acc.append(avg_acc.tolist())
    return np.array(this_acc).T.tolist()
    
def batch_acc_from_joint_angles(dt_joints_pos, gt_joints_pos, thresholds):
    # dt_joitns_pos, gt_joints_pos : B x 7 x 1
    this_acc = []
    #dist_angles = np.abs(dt_joints_pos,gt_joints_pos)[:, :-1, 0] # B x 7 Find the bug！
    dist_angles = np.abs(dt_joints_pos - gt_joints_pos)[:, :, 0]#[:, :, 0] # B x   
    #print("dist_angles", dist_angles)
    for thresh in thresholds:
        radian_thresh = math.radians(thresh)
        avg_acc = np.mean(dist_angles < radian_thresh, axis=-1) 
        #print(avg_acc.shape)
        #print(avg_acc)
        this_acc.append(avg_acc.tolist())
    #print(np.array(this_acc).T.shape)
    return np.array(this_acc).T.tolist()

def add_metrics(add, add_auc_threshold=0.1):
    # add list
    add = [x for x in add if math.isnan(x) == False] 
    
    add = np.array(add)
    mean_add = np.mean(add)
    median_add = np.median(add)
    std_add = np.std(add)
    length_add = len(add)
    
    #print(add)
    
#    assert np.nan in add
#    print("Find nan")
    
    # Compute Integral via Approximation Algorithms
    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, add_auc_threshold, delta_threshold)
    counts = []
    for value in add_threshold_values:
        under_threshold = len(np.where(add <= value)[0]) / float(
            length_add
        )
        counts.append(under_threshold)

    auc = np.trapz(counts, dx=delta_threshold) / float(add_auc_threshold)
    metrics = {
        "add_mean": mean_add,
        "add_median": median_add,
        "add_std": std_add,
        "add_auc": auc,
        "add_auc_thresh": add_auc_threshold,
    }
    return metrics

def batch_pck_from_pose(keypoints_gt, keypoints_dt):
    # keypoints_gt : B x N x 2
    # keypoints_detected : B x N x 2
    kp_2d_errors = keypoints_dt - keypoints_gt # B x N x 2
    kp_2d_l2_errors = np.linalg.norm(kp_2d_errors, axis=-1) # B x N
    kp_2d_l2_errors_mean = np.mean(kp_2d_l2_errors, axis=1) # B
    return kp_2d_l2_errors_mean.tolist()

def batch_1d_pck_from_pose(keypoints_gt, keypoints_dt):
    # keypoints_gt : B x N x 1
    # keypoints_dt : B x N x 1
    kp_1d_errors = keypoints_dt - keypoints_gt
    kp_1d_l2_errors = np.linalg.norm(kp_1d_errors, axis=-1) # B x N
    kp_1d_l2_errors_mean = np.mean(kp_1d_l2_errors, axis=1) # B
    return kp_1d_l2_errors_mean.tolist()

def pck_metrics(kp_l2_errors, auc_pixel_threshold=12.0):
    if len(kp_l2_errors) > 0:
        kp_l2_errors = np.array(kp_l2_errors)
        kp_l2_error_mean = np.mean(kp_l2_errors)
        kp_l2_error_median = np.median(kp_l2_errors)
        kp_l2_error_std = np.std(kp_l2_errors)
        
        # compute the auc
        delta_pixel = 0.01
        pck_values = np.arange(0, auc_pixel_threshold, delta_pixel)
        y_values = []

        for value in pck_values:
            valids = len(np.where(kp_l2_errors < value)[0])
            y_values.append(valids)

        kp_auc = (
            np.trapz(y_values, dx=delta_pixel)
            / float(auc_pixel_threshold)
            / float(len(kp_l2_errors))
        )
    else:
        kp_l2_error_mean = None
        kp_l2_error_median = None
        kp_l2_error_std = None
        kp_auc = None
    
    metrics = {
        "l2_error_mean_px": kp_l2_error_mean,
        "l2_error_median_px": kp_l2_error_median,
        "l2_error_std_px": kp_l2_error_std,
        "l2_error_auc": kp_auc,
        "l2_error_auc_thresh_px": auc_pixel_threshold,
    }
    return metrics

    
    
    
    

def print_to_screen_and_file(file, text):
    print(text)
    file.write(text + "\n")





















