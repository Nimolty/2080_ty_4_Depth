#import _init_paths
import argparse
import random
import time
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import torch.utils.data
import torchvision.transforms as transforms
import torchgeometry as tgm
import torch.nn as nn

from depth_c2rp.REDE.lib.KNN_CUDA.knn_cuda import KNN
from depth_c2rp.REDE.lib.loss import calculate_error, batch_least_square
from depth_c2rp.REDE.lib.transformations import quaternion_matrix, quaternion_from_matrix
import warnings
warnings.filterwarnings("ignore")

def compute_rede_rt(model_kp, points_pred):
    # model_kp : 1 x num_kps x 3, robot space 3d
    # points_pred : 1 x num_kps x 3, camera space 3d
    B, num_kps, _ = model_kp.shape
    device = model_kp.device
    all_index = torch.combinations(torch.arange(num_kps), 3)
    all_r, all_t = batch_least_square(model_kp.squeeze()[all_index, :], points_pred.squeeze()[all_index, :], torch.ones([all_index.shape[0], 3]).to(device))
    
    all_e = calculate_error(all_r, all_t, model_kp,  points_pred)
    e = all_e.unsqueeze(0).unsqueeze(2)
    w = torch.softmax(1 / e, 1).squeeze().unsqueeze(1)
    all_qua = tgm.rotation_matrix_to_quaternion(torch.cat((all_r, torch.tensor([0., 0., 1.]).to(device).unsqueeze(1).repeat(all_index.shape[0], 1, 1)), dim=2))
    pred_qua = torch.sum(w * all_qua, 0)
    pred_r = pred_qua.view(1, 1, -1)
    bs, num_p, _ = pred_r.size()
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
    pred_r = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                        (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                        (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                        (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1), \
                        (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                        (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                        (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                        (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                        (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
    my_r = pred_r.squeeze()
    my_t = torch.sum(w * all_t, 0)
    
    pose_pred = torch.eye(4)[None, :, :].to(device)
    pose_pred[0, :3, :3] = my_r
    pose_pred[0, :3, 3] = my_t
    
    return pose_pred