from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import os
import numpy as np
import json
import torch.nn.functional as F
import glob
import cv2
import matplotlib.pyplot as plt
import json

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
    

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)


def load_model(model, model_path, lr, optimizer=None,resume=True):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
   
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  # print("model_state_dict", model_state_dict)
  for k in state_dict:
    if k in model_state_dict:
      if (state_dict[k].shape != model_state_dict[k].shape):
        if resume:
          print('Reusing parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          if state_dict[k].shape[0] < state_dict[k].shape[0]:
            model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
          else:
            model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
          state_dict[k] = model_state_dict[k]
        else:
          print('Skip loading parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k))
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      # optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True


def visualize_training_loss(losses, writer, batch_idx, epoch, data_loader_length):
    # expect losses are dict(), and values are tensor
    for key, value in losses.items():
        this_loss = value.item()
        writer.add_scalar(f"Loss/{key}_loss", this_loss, batch_idx + (epoch-1) * data_loader_length)


def visualize_validation_loss(val_loss, val_mask_loss, val_3d_loss, val_pos_loss, val_rt_loss, writer, epoch):
    writer.add_scalar(f"Val/Total_Loss", np.mean(val_loss), epoch-1)
    writer.add_scalar(f"Val/Mask_Loss", np.mean(val_mask_loss), epoch-1)
    writer.add_scalar(f"Val/3D_Loss", np.mean(val_3d_loss), epoch-1)
    writer.add_scalar(f"Val/Pos_Loss", np.mean(val_pos_loss), epoch-1)
    writer.add_scalar(f"Val/Rt_Loss", np.mean(val_rt_loss), epoch-1)

def visualize_training_masks(masks, writer, device, batch_idx, epoch, data_loader_length):
    # masks为BxCxHxW 这里的C为model_classes
    B, C, H, W = masks.shape
    new_masks = nn.functional.softmax(masks, dim=1)
    mask_argmax = new_masks.argmax(dim=1)
    #print("mask_argmax", mask_argmax.shape)
    #print("mask_argmax", torch.where(mask_argmax == 0)[0].shape)
    #print("new_masks", new_masks.shape)
    res = torch.zeros(B, H, W*C).to(device)
#    for i in range(C):
#        res[:, :, i*W : (i+1) * W] = new_masks[:, i, :, :]
    for i in range(C):
        res[:, :, i*W:(i+1)*W][torch.where(mask_argmax == i)] = 1
    res = res[:, None, :, :]
    #print(torch.where(res != 0)[0].shape)
    #print("res.shape", res.shape)
    grid_image = make_grid(res * 255, B//4, normalize=False, scale_each=False)
    #print("grid_image", grid_image.shape)
    writer.add_image(f'dt_masks', grid_image.detach().cpu().numpy(), batch_idx + (epoch-1) * data_loader_length, dataformats='CHW')

def visualize_training_lr(lr, writer, batch_idx, epoch, data_loader_length):
    writer.add_scalar(f"LR/Learning_Rate", lr, batch_idx + (epoch-1) * data_loader_length)


def visualize_inference_results(add_res, mAP_res, writer, epoch):
    for key, value in add_res.items():
        writer.add_scalar(f"Add/{key}", value, epoch)
    for key, value in mAP_res.items():
        writer.add_scalar("mAP/{:.5f} m".format(float(key)), value, epoch)
    
 
def find_seq_data_in_dir(input_dir):
    # input_dir例如这里的 "/DATA/disk1/hyperplane/Depth_C2RP/Data/Ty_trial/"
    input_dir = os.path.expanduser(input_dir)
    assert os.path.exists(
    input_dir
    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = glob.glob(os.path.join(input_dir, "0*"))
    print(len(dirlist))
    
    found_data = []
    for each_dir in dirlist:
        found_data_this_video = []
        output_dir = os.path.join(input_dir, each_dir)
        
        image_exts_to_try = ["_color.png", "_meta.json", "_mask.exr", "_simDepthImage.exr"]
        
        img_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(image_exts_to_try[0])]
        meta_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(image_exts_to_try[1])]
        mask_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(image_exts_to_try[2])]
        simDepth_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(image_exts_to_try[3])]
        
        img_paths.sort()
        meta_paths.sort()
        mask_paths.sort()
        simDepth_paths.sort()
        
        min_num_image = np.min([len(img_paths), len(meta_paths), len(mask_paths), len(simDepth_paths)])
        if min_num_image == 0 or min_num_image == 1:
            # print("Expected the Data Length >= 2")
            continue
        else:
            pass
        
        assert min_num_image >= 2
        
        img_paths = img_paths[:min_num_image]
        meta_paths = meta_paths[:min_num_image]
        mask_paths = mask_paths[:min_num_image]
        simDepth_paths = simDepth_paths[:min_num_image]        
        
        scene_name = meta_paths[0].split("/")[-2]
        frame_names = [f.split("/")[-1][:4] for f in meta_paths]
        for idx in range(min_num_image):
            assert frame_names[idx] == img_paths[idx].split("/")[-1][:4]
            assert frame_names[idx] == meta_paths[idx].split("/")[-1][:4]
            assert frame_names[idx] == mask_paths[idx].split("/")[-1][:4]
            assert frame_names[idx] == simDepth_paths[idx].split("/")[-1][:4]
            
            if idx == min_num_image - 1:
                continue
            
            if int(frame_names[idx+1]) - int(frame_names[idx]) > 1:
                continue
            
            this_seq = {}
            this_seq['prev_frame_name'] = os.path.join(scene_name, frame_names[idx])
            this_seq["prev_frame_img_path"] = img_paths[idx]
            this_seq["prev_frame_data_path"] = meta_paths[idx]
            this_seq["prev_frame_mask_path"] = mask_paths[idx]
            this_seq["prev_frame_simdepth_path"] = simDepth_paths[idx]
            this_seq['next_frame_name'] = os.path.join(scene_name, frame_names[idx+1])
            this_seq["next_frame_img_path"] = img_paths[idx+1]
            this_seq["next_frame_data_path"] = meta_paths[idx+1]
            this_seq["next_frame_mask_path"] = mask_paths[idx+1]
            this_seq["next_frame_simdepth_path"] = simDepth_paths[idx+1]
            found_data_this_video.append(this_seq)
        
        found_data = found_data + found_data_this_video
        # print('found_data_this_video', found_data_this_video)
    
    return found_data


def load_keypoints_and_joints(json_path, keypoint_names, joint_names):
    json_in = open(json_path, 'r')
    json_data = json.load(json_in)[0]
    keypoints_data = json_data["keypoints"]
    joints_data = json_data["joints"]
    
    keypoints_res = {
                    "Projection" : [],
                    "Part_ID" : [],
                    "Location_wrt_cam" : [],
                    "O2C_wxyz" : [],
                    "Scale" : [],
                    "Center": []
                    }
    joints_res = {
                 "Angle" : [],
                 "Location_wrt_cam" : []
                 }
    
    for idx, kp_name in enumerate(keypoint_names):
        assert kp_name == keypoints_data[idx]["Name"], \
        "Expected keypoint '{}' in datafile '{}', but receive '{}' ".format(
        kp_name, json_path, keypoints_data["Name"]
        )
        keypoints_res["Projection"].append(keypoints_data[idx]["projected_location"])
        keypoints_res["Part_ID"].append(keypoints_data[idx]["Part ID"])
        keypoints_res["Location_wrt_cam"].append(keypoints_data[idx]["location_wrt_cam"])
        
        O2C_mat = torch.from_numpy(np.array(keypoints_data[idx]["R2C_mat"]))
        O2C_wxyz = matrix_to_quaternion(O2C_mat).numpy().tolist()
        keypoints_res["O2C_wxyz"].append(O2C_wxyz)
        
        
        keypoints_res["Scale"].append(keypoints_data[idx]["scale"])
        # keypoints_res["Center"].append(keypoints_data[idx]["center_loc"])
    for idx, joint_name in enumerate(joint_names):
        assert joint_name == joints_data[idx]["Name"], \
        "Expected joint '{}' in datafile '{}', but receive '{}' ".format(
        joint_name, json_path, joints_data[idx]["Name"]
        )
        joints_res["Angle"].append(joints_data[idx]["position"])
        joints_res["Location_wrt_cam"].append(joints_data[idx]["location_wrt_cam"])
    
    return keypoints_res, joints_res


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))  
        
 
 
def quaternionToRotation(q):
    w, x, y, z = q
    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y - 2 * w * z
    r02 = 2 * x * z + 2 * w * y

    r10 = 2 * x * y + 2 * w * z
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z - 2 * w * x

    r20 = 2 * x * z - 2 * w * y
    r21 = 2 * y * z + 2 * w * x
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2
    r = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    return r


def batch_quaternion_matrix(pr):
    # pr :[4, B]
    R = torch.cat(((1.0 - 2.0 * (pr[2, :] ** 2 + pr[3, :] ** 2)).unsqueeze(dim=1), \
                  (2.0 * pr[1, :] * pr[2, :] - 2.0 * pr[0, :] * pr[3, :]).unsqueeze(dim=1), \
                  (2.0 * pr[0, :] * pr[2, :] + 2.0 * pr[1, :] * pr[3, :]).unsqueeze(dim=1), \
                  (2.0 * pr[1, :] * pr[2, :] + 2.0 * pr[3, :] * pr[0, :]).unsqueeze(dim=1), \
                  (1.0 - 2.0 * (pr[1, :] ** 2 + pr[3, :] ** 2)).unsqueeze(dim=1), \
                  (-2.0 * pr[0, :] * pr[1, :] + 2.0 * pr[2, :] * pr[3, :]).unsqueeze(dim=1), \
                  (-2.0 * pr[0, :] * pr[2, :] + 2.0 * pr[1, :] * pr[3, :]).unsqueeze(dim=1), \
                  (2.0 * pr[0, :] * pr[1, :] + 2.0 * pr[2, :] * pr[3, :]).unsqueeze(dim=1), \
                  (1.0 - 2.0 * (pr[1, :] ** 2 + pr[2, :] ** 2)).unsqueeze(dim=1)),
                  dim=1).contiguous().view(-1, 3, 3)  # [nv, 3, 3]
    return R
 
 
def check_input(batch, cfg):
    #prev_img, next_img = batch["prev_frame_img_as_input"][0].numpy(), batch["next_frame_img_as_input"][0].numpy()
    #prev_img, next_img = cv2.cvtColor(prev_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR), cv2.cvtColor(next_img.transpose(1,2,0), cv2.COLOR_RGB2BGR)
    next_img =  batch["next_frame_img_as_input"][0].numpy()
    next_img =  cv2.cvtColor(next_img.transpose(1,2,0), cv2.COLOR_RGB2BGR)
    #cv2.imwrite(f"./check/prev_cropped_img.png", (prev_img*0.5+0.5)*255)
    cv2.imwrite(f"./check/next_cropped_img.png", (next_img*0.5+0.5)*255)
    # prev_mask, next_mask = batch["prev_frame_mask_as_input"][0].numpy(), batch["next_frame_mask_as_input"][0].numpy()
    next_mask =  batch["next_frame_mask_as_input"][0].numpy()
    H, W = next_mask.shape
    for idx in range(cfg["MODEL"]["MODEL_CLASSES"]):
        #prev_mask_out, next_mask_out = np.zeros((H, W)), np.zeros((H, W))
        next_mask_out = np.zeros((H, W))
        print("next_mask", np.where(next_mask==idx))
        #prev_mask_out[np.where(prev_mask==idx)]=1
        next_mask_out[np.where(next_mask==idx)]=1
        #cv2.imwrite(f"./check/prev_cropped_mask_{idx}.png", prev_mask_out[:, :, None]*255)
        cv2.imwrite(f"./check/next_cropped_mask_{idx}.png", next_mask_out[:, :, None]*255)
    
    #prev_simdepth, next_simdepth = batch["prev_frame_simdepth_as_input"][0].numpy(),  batch["next_frame_simdepth_as_input"][0].numpy()
    next_simdepth = batch["next_frame_simdepth_as_input"][0].numpy()
    #prev_xy_wrt_cam = batch["prev_frame_xy_wrt_cam"][0].numpy()
    #prev_uv = batch["prev_frame_uv"][0].numpy()
    #prev_normals_crop = batch["prev_normals_crop"][0].numpy()
    next_xy_wrt_cam = batch["next_frame_xy_wrt_cam"][0].numpy()
    next_uv = batch["next_frame_uv"][0].numpy()
    print("next_uv", next_uv)
    next_normals_crop = batch["next_normals_crop"][0].numpy()
    
    #prev_xyzn = np.concatenate([prev_xy_wrt_cam, prev_simdepth, prev_normals_crop], axis=0).reshape(6, -1)
    next_xyzn = np.concatenate([next_xy_wrt_cam, next_simdepth, next_normals_crop], axis=0).reshape(6, -1)
    #np.savetxt(f"./check/prev_xyzn.txt", prev_xyzn.T)
    np.savetxt(f"./check/next_xyzn.txt", next_xyzn.T)
    #plt.imsave(f"./check/prev_cropped_depth.png", prev_simdepth[0])
    plt.imsave(f"./check/next_cropped_depth.png", next_simdepth[0])
    #prev_frame_joints_wrt_cam, next_frame_joints_wrt_cam = batch["prev_frame_joints_wrt_cam"][0].numpy(), batch["next_frame_joints_wrt_cam"][0].numpy()
    next_frame_joints_wrt_cam = batch["next_frame_joints_wrt_cam"][0].numpy()
    #prev_base_trans, next_base_trans = batch["prev_frame_base_trans"][0].numpy(), batch["next_frame_base_trans"][0].numpy()
    #prev_base_quat, next_base_quat = batch["prev_frame_base_quaternion"][0].numpy(), batch["next_frame_base_quaternion"][0].numpy()
    next_base_trans = batch["next_frame_base_trans"][0].numpy()
    next_base_quat = batch["next_frame_base_quaternion"][0].numpy()
    #np.savetxt(f"./check/prev_joints_wrt_cam.txt", prev_frame_joints_wrt_cam)
    np.savetxt(f"./check/next_joints_wrt_cam.txt", next_frame_joints_wrt_cam)
    #np.savetxt(f"./check/prev_base_trans.txt", prev_base_trans)
    np.savetxt(f"./check/next_base_trans.txt", next_base_trans)
    #np.savetxt(f"./check/prev_base_quat.txt", prev_base_quat)
    np.savetxt(f"./check/next_base_quat.txt", next_base_quat)
    print("next_frame_img_path", batch["next_frame_img_path"][0])

def load_camera_intrinsics(camera_data_path):

    # Input argument handling
    assert os.path.exists(
        camera_data_path
    ), 'Expected path "{}" to exist, but it does not.'.format(camera_data_path)

    # Create YAML/json parser
    json_in = open(camera_data_path, 'r')
    cam_settings_data = json.load(json_in)
    

    camera_fx = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fx"]
    camera_fy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fy"]
    camera_cx = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cx"]
    camera_cy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cy"]
    camera_K = np.array(
        [[camera_fx, 0.0, camera_cx], [0.0, camera_fy, camera_cy], [0.0, 0.0, 1.0]]
    )

    return camera_K
    
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
 
def compute_concat_loss(R, T, Angles, device):
    # R : B x 3 x 3
    # T : B x 3 x 1
    # Angles : B x N
    
    B, N, _ = Angles.shape
    ori_trans_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.333], [0.0,0.0,0.0],\
                                        [0.0,-0.316,0.0], [0.0825,0.0,0.0],[-0.0825,0.384,0.0],\
                                        [0.0,0.0,0.0], [0.088,0.0,0.0],[0.0,0.0,0.107],\
                                        [0.0,0.0,0.0584], [0.0,0.0,0.0]
                                        ])).float().to(device)
    ori_angles_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
              [1.5707963267948966,0.0,0.0], [1.5707963267948966,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
               [1.5707963267948966,0.0,0.0],  [1.5707963267948966,0.0,0.0], [0.0,0.0,-1.5707963267948966/2],\
               [0.0,0.0,0.0], [0.0,0.0,0.0]
                                        ])).float().to(device)
    joints_info = [
              {"base" : 0, "offset" : [0.0, 0.0, 0.14]},
              {"base" : 1, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 3, "offset" : [0.0, 0.0, -0.1210]},
              {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 5, "offset" : [0.0, 0.0, -0.2590]},
              {"base" :5, "offset" : [0.0, 0.0158, 0.0]},
              {"base" : 7, "offset" : [0.0, 0.0, 0.0520]},
              {"base" : 8, "offset" : [0.0, 0.0, 0.0584]}
              ]
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    #assert N == ori_trans_list.shape[0], f"{ori_trans_list.shape} and {N}"
    
    kps_list, R2C_list = [], []
    for j in range(ori_trans_list.shape[0]):
        if j == 0:
            kps_list.append(ori_trans.clone())
            R2C_list.append(ori_mat.clone())
            continue
        this_mat = ori_angles[j].repeat(B, 1).reshape(B, 3, 3)
        trans = ori_trans_list[j].unsqueeze(0).repeat(B, 1)
        if 1 <= j <= 7:
            new_mat = _axis_angle_rotation("Z", Angles[:, j-1, 0])
            #print("this_mat.shape", this_mat.shape)
            #print("new_mat.shape", new_mat.shape)
            this_mat = torch.bmm(this_mat, new_mat)
        if j == 9:
            trans = trans + Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
        if j == 10:
            trans = trans + (-2) * Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
         
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        kps_list.append(ori_trans.float().clone())
        R2C_list.append(ori_mat.float().clone())
    
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
    for idx in range(len(joints_info)):
        base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"]))[None, :, None].repeat(B, 1, 1).to(device)
        this_x3d = torch.bmm(R2C_list[base_idx], offset.float()) + kps_list[base_idx]
        joints_x3d_rob[:, idx, :, :] = this_x3d.clone()
    
    joints_x3d_rob = joints_x3d_rob.squeeze(3)
    joints_x3d_cam= torch.bmm(R, joints_x3d_rob.permute(0, 2, 1).contiguous()) + T
    return joints_x3d_cam.permute(0,2,1).contiguous()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
 
 
 
 
 
 
 
 
 
 
 
 

