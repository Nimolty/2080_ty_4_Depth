import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

import cv2
import numpy as np
import copy
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def to_colormap(heatmap_tensor, device, cmap = 'jet', cmap_range=(None, None)):
    """Converts a heatmap into an image assigning to the gaussian values a colormap.

    Parameters
    ----------
    heatmap_tensor: torch.Tensor
        Heatmap as a tensor (NxHxW).
    cmap: str
        Colormap to use for heatmap visualization.
    cmap_range: tuple
        Range of values for the colormap.

    Returns
    -------
    output: np.array
        Array of N images representing the heatmaps.
    """
    if not isinstance(heatmap_tensor, np.ndarray):
        try:
            heatmap_tensor = heatmap_tensor.to('cpu').numpy()
        except RuntimeError:
            heatmap_tensor = heatmap_tensor.detach().to('cpu').numpy()

    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])

    heatmap_tensor = np.max(heatmap_tensor, axis=1)

    output = []
    batch_size = heatmap_tensor.shape[0]
    for b in range(batch_size):
        rgb = cmap.to_rgba(heatmap_tensor[b])[:, :, :-1]
        output.append(rgb[:, :, ::-1])  # RGB to BGR

    output = np.asarray(output).astype(np.float32)
    output = output.transpose(0, 3, 1, 2)  # (b, h, w, c) -> (b, c, h, w)

    return output

def random_blend_grid(true_blends, pred_blends):
    """Stacks predicted and ground truth blended images (heatmap+image) by column.

    Parameters
    ----------
    true_blends: np.array
        Ground truth blended images.
    pred_blends: np.array
        Predicted blended images.

    Returns
    -------
    grid: np.array
        Grid of predicted and ground truth blended images.
    """
    grid = []
    for i in range(0, len(true_blends)):
        grid.append(np.concatenate(true_blends[i], axis=2))
        grid.append(np.concatenate(pred_blends[i], axis=2))
    return grid

def get_joint_3d_pred(heatmap_pred, cfg, h, w, c, input_K):
    if cfg["MODEL"]["NAME"] == "stacked_hourglass":
        heatmap_pred = (F.interpolate(heatmap_pred[-1], (h, w), mode='bicubic', align_corners=False) + 1) / 2.
    else:
        heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
    
    B, C, H, W = heatmap_pred.shape
    joints_3d_pred = np.ones((B, C // 2, 3))
    joints_3d_pred[:, :, :2] = np.array([np.array(
        [np.unravel_index(np.argmax(heatmap_pred[b, i, :, :].detach().cpu().numpy(),
                                    axis=None), heatmap_pred[b, i, :, :].shape)
         for i in range(0, c // 2)])
        for b in range(heatmap_pred.shape[0])])[..., ::-1]
    # add Z coordinate from UZ heatmap
    z = np.array([np.array(
        [np.unravel_index(np.argmax(heatmap_pred[b, i, :, :].detach().cpu().numpy(),
                                    axis=None), heatmap_pred[b, i, :, :].shape)
         for i in range(c // 2, c)])
        for b in range(heatmap_pred.shape[0])])[..., :1]
    Z_min, _, dZ = cfg["DATASET"]["DEPTH_RANGE"]
    z = ((z * dZ) + Z_min) / 1000
    # convert 2D predicted joints to 3D coordinate multiplying by inverse intrinsic matrix
    inv_intrinsic = torch.inverse(input_K).unsqueeze(1).repeat(1,
                                                               joints_3d_pred.shape[1],
                                                               1,
                                                               1).detach().cpu().numpy()
    joints_3d_pred = (inv_intrinsic @ joints_3d_pred[:, :, :, None]).squeeze(-1)
    joints_3d_pred *= z
    return heatmap_pred, joints_3d_pred

def get_blended_images(gt_images, K, joints_3d_gt, joints_3d_pred, device, heatmaps_pred, heatmaps_gt):
    pred_images = gt_images.copy()
    blend_images = gt_images.copy()   
    
    joints_2d_gt = (K[:, None, :, :] @ joints_3d_gt[..., None]).squeeze(-1)
    joints_2d_gt = joints_2d_gt / joints_2d_gt[..., 2:]
    joints_2d_pred = (K[:, None, :, :] @ joints_3d_pred[..., None]).squeeze(-1)
    joints_2d_pred = (joints_2d_pred / (joints_2d_pred[..., 2:] + 1e-9))
    for b in range(len(gt_images)):
        for joint_2d_gt, joint_2d_pred in zip(joints_2d_gt[b, ...], joints_2d_pred[b, ...]):
            cv2.circle(gt_images[b],
            (int(joint_2d_gt[0]), int(joint_2d_gt[1])), 2, (255, 0, 0), -1)
            cv2.circle(pred_images[b],
            (int(joint_2d_pred[0]), int(joint_2d_pred[1])), 2, (0, 255, 0), -1)
    gt_images = np.stack(gt_images).transpose(0, 3, 1, 2).astype(float) / 255.
    pred_images = np.stack(pred_images).transpose(0, 3, 1, 2).astype(float) / 255.
    c = heatmaps_pred.shape[1]
    blend_images = np.stack(blend_images).transpose(0, 3, 1, 2).astype(float) / 255.
    pred_blend_uv = 0.5 * blend_images + 0.5 * to_colormap(heatmaps_pred[:8, :(c // 2)], device)
    pred_blend_uz = to_colormap(heatmaps_pred[:8, (c // 2):], device)
    true_blend_uv = 0.5 * blend_images + 0.5 * to_colormap(heatmaps_gt[:8, :(c // 2)], device)
    true_blend_uz = to_colormap(heatmaps_gt[:8, (c // 2):], device)
    
    return gt_images, pred_images, true_blend_uv, true_blend_uz, pred_blend_uv, pred_blend_uz
    
def log_and_visualize_single(log_writer, global_iter,
                      gt_results=None, pred_results=None,
                      true_blends_UV=None, pred_blends_UV=None,
                      true_blends_UZ=None, pred_blends_UZ=None):
    # plot and visualize per keypoint precision histogram
    if gt_results is not None and pred_results is not None:
        results_grid = random_blend_grid(gt_results, pred_results)
        results_grid = np.concatenate(results_grid, axis=1)
        results_grid = torch.from_numpy(results_grid)
        log_writer.add_image(tag=f'Train/Joints Prediction', img_tensor=results_grid,
                                  global_step=global_iter)

    # visualize pred and gt UV heatmaps
    if true_blends_UV is not None and pred_blends_UV is not None:
        eval_grid = random_blend_grid(true_blends_UV, pred_blends_UV)
        eval_grid = np.concatenate(eval_grid, axis=1)
        eval_grid = torch.from_numpy(eval_grid)
        log_writer.add_image(tag=f'Train/Joints UV heatmaps', img_tensor=eval_grid,
                                  global_step=global_iter)

    # visualize pred and gt UZ heatmaps
    if true_blends_UZ is not None and pred_blends_UZ is not None:
        eval_grid = random_blend_grid(true_blends_UZ, pred_blends_UZ)
        eval_grid = np.concatenate(eval_grid, axis=1)
        eval_grid = torch.from_numpy(eval_grid)
        log_writer.add_image(tag=f'Train/Joints UZ heatmaps', img_tensor=eval_grid,
                                  global_step=global_iter)

    
    
    
    
    
    
    
    
