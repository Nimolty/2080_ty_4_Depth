import numpy as np
import os
import torch
import cv2
from depth_c2rp.utils.utils import quaternionToRotation


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

def add_metrics(add, add_auc_threshold=0.1):
    # add np.ndarray
    mean_add = np.mean(add)
    median_add = np.median(add)
    std_add = np.std(add)
    length_add = len(add)
    
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

def print_to_screen_and_file(file, text):
    print(text)
    file.write(text + "\n")





















