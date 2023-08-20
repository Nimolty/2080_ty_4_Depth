import numpy as np
import torch
import json
import glob
import os
import random
import torch.nn.functional as F
import pickle5 as pickle
from tqdm import tqdm
import time

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

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

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

def load_dream_data(data_path):
    json_in = open(data_path, 'r')
    json_data = json.load(json_in)
    
    quaternion_xyzw = json_data["objects"][0]["quaternion_xyzw"]
    joint_angles = json_data["sim_state"]['joints']
    joint_angles = [j["position"] for j in joint_angles][:7]
    
    quaternion_wxyz = quaternion_xyzw[-1:] + quaternion_xyzw[:-1]
    translation = np.array(json_data["objects"][0]["location"]) * 0.01
    
    return quaternion_wxyz, translation, joint_angles

def compute_kps_joints_loss(R, T, Angles, device):
    # R : B x 3 x 3
    # T : B x 3 x 1
    # Angles : B x 7 x 1
    B, _, _ = Angles.shape
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
              # joint1 and plus 3 pts
              {"base" : 0, "offset" : [0.0, 0.0, 0.14]},
              # joint2 and plus 1 pt
              {"base" : 1, "offset" : [0.0, 0.0, 0.0]},
              # joint3 and plus 1 pt
              {"base" : 3, "offset" : [0.0, 0.0, -0.1210]},
              # joint4 and plus 1 pt
              {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
              # joint5 and plus 1 pt
              {"base" : 5, "offset" : [0.0, 0.0, -0.2590]},
              # joint6 and plus 1 pt
              {"base" :5, "offset" : [0.0, 0.0158, 0.0]},
              # joint7 and plus 1 pt
              {"base" : 7, "offset" : [0.0, 0.0, 0.0520]},
              ]
    
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    
    
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
            this_mat = torch.bmm(this_mat, new_mat)
         
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        kps_list.append(ori_trans.float().clone())
        R2C_list.append(ori_mat.float().clone())
        
    N = 9
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
    for idx in range(N):
        joints_x3d_rob[:, idx, :, :] = kps_list[idx]
    
    
    joints_x3d_rob = joints_x3d_rob.squeeze(3)
    joints_x3d_cam= torch.bmm(R, joints_x3d_rob.permute(0, 2, 1).contiguous()) + T
    return joints_x3d_cam.permute(0,2,1).contiguous()

def compute_3n_loss_42_cam(R, T, Angles, device=torch.device("cpu")):
    # R : B x 3 x 3
    # T : B x 3 x 1
    # Angles : B x N x 1
    B, M, _ = Angles.shape
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
              # base and plus 3 pts
              {"base" : 0, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 0, "offset" : [0.0, 0.0, 0.14]}, # panda_joint1
              {"base" : 0, "offset" : [-0.11, 0.0, 0.0]},
              
              
              {"base" : 1, "offset" : [0.0, 0.0, 0.0]}, # panda_joint2
              {"base" : 1, "offset" : [0.0, -0.1294, -0.0]}, 
              
              
              {"base" : 2, "offset" : [0.0, -0.1940, 0.0]}, # panda_joint3 
              
              {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
              {"base" : 4, "offset" : [0.0, 0.0, 0.1111]},
              {"base" : 4, "offset" : [0.0, 0.1240, 0.0]},
              
              
              {"base" : 5, "offset" : [0.0, 0.1299, 0.0]},
              
              
              {"base" :6, "offset" : [0.0, 0.0, 0.0583]},
              {"base" :6, "offset" : [0.088, 0.0, 0.0]}, 
              
              {"base" : 7, "offset" : [0.0, 0.0, 0.1520]},
              {"base" : 7, "offset" : [0.06, 0.06, 0.1520]}
              ]
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    
    
    kps_list, R2C_list = [], []
    for j in range(11):
        if j == 0:
            kps_list.append(ori_trans.clone())
            R2C_list.append(ori_mat.clone())
            continue
        this_mat = ori_angles[j].repeat(B, 1).reshape(B, 3, 3)
        trans = ori_trans_list[j].unsqueeze(0).repeat(B, 1)
        if 1 <= j <= 7:
            new_mat = _axis_angle_rotation("Z", Angles[:, j-1, 0])
            this_mat = torch.bmm(this_mat, new_mat)         
         
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        kps_list.append(ori_trans.float().clone())
        R2C_list.append(ori_mat.float().clone())


        
    N = len(joints_info)    
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)
    for idx in range(len(joints_info)):
        base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"]))[None, :, None].repeat(B, 1, 1).to(device)
        this_x3d = torch.bmm(R2C_list[base_idx], offset.float()) + kps_list[base_idx]
        joints_x3d_rob[:, idx, :, :] = this_x3d.clone()
    
    joints_x3d_rob = joints_x3d_rob.squeeze(3)
    joints_x3d_cam= torch.bmm(R, joints_x3d_rob.permute(0, 2, 1).contiguous()) + T
    return joints_x3d_cam.permute(0,2,1).contiguous() 

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_K = load_camera_intrinsics("/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_dream_T7/panda_synth_train_dr/_camera_settings.json")
    #json_paths_list = [f"/mnt/hyperplane/yangtian/synthetic/Panda/panda_synth_train_dr/20995{i}.json" for i in range(1, 10, 2)]
    t1 = time.time()
    print("t1 ", t1)
    #a = os.listdir("/mnt/hyperplane/yangtian/synthetic/Panda/panda_synth_train_dr/")
    t2 = time.time()
    #print(len(a))
    print("t2 - t1", t2 - t1)
    
    json_paths_list = glob.glob(os.path.join("/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_dream_T7/panda_synth_train_dr/", "*jpg"))
    json_paths_list.sort()
    #print(json_paths_list)
    json_paths_list = json_paths_list[:-2]

    for json_path in tqdm(json_paths_list):
        root_dir = "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_dream_T7/panda_synth_train_dr/"
        exists_or_mkdir(root_dir)
        json_idx = json_path.split('/')[-1][:6]
        path_meta = os.path.join(root_dir, f"{json_idx}.pkl")
        
        if os.path.exists(path_meta):
            continue
        quaternion, translation, this_joint_angles = load_dream_data(json_path)
        
        # joint angles 
        this_joint_angles = torch.from_numpy(np.array(this_joint_angles)).float().to(device).reshape(1, 7, 1) # 1 x 7 x 1
        this_joint_angles_np = this_joint_angles.detach().cpu().numpy()[0]
        # quaternion
        this_quaternion = torch.from_numpy(np.array(quaternion)).float().to(device).reshape(1, 4)
        R_NORMAL_UE = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ])
        R_NORMAL_UE = torch.from_numpy(R_NORMAL_UE).float().to(device)
        this_rot = quaternion_to_matrix(this_quaternion) 
        this_rot = this_rot @ R_NORMAL_UE  # 1 x 3 x 3
        
        # translation 
        this_translation = torch.from_numpy(np.array(translation)).float().to(device).reshape(1, 3, 1) # 1 x 3 x 1
        
        # generate 7 link kps
        kps_9_cam_tensor = compute_kps_joints_loss(this_rot, 
                                                   this_translation,  
                                                   this_joint_angles, 
                                                   device)[0] # 9 x 3
        kps_9_cam_np = kps_9_cam_tensor.detach().cpu().numpy()
                
        # generate 14 kps
        kps_14_cam_tensor = compute_3n_loss_42_cam(this_rot,
                                                   this_translation,  
                                                   this_joint_angles, 
                                                   device)[0] # 14 x 3
        kps_14_cam_np = kps_14_cam_tensor.detach().cpu().numpy()
        
        kps_7_cam_np = kps_9_cam_np[[0,2,3,4,6,7,8]]
        
        link_names = ["Link0","Link1","Link2","Link3","Link4","Link5","Link6","Link7","panda_hand"]
        joint_7_names = [f"panda_joint{idx}" for idx in range(1, 8)]
        joint_14_names = [f"panda_joint_3n_{idx}" for idx in range(1, 15)]
    
        meta_json = []
        meta_dict = {}
        meta_dict["SCENE_NUM"] = json_idx
        meta_dict["FRAME NUM"] = json_idx
        meta_dict["ROBOT NAME"] = "Franka_Emika_Panda"
        meta_dict["keypoints"] = []
        meta_dict["joints"] = []
        meta_dict["joints_3n_fixed_42"] = []
        
        for idx, key in enumerate(link_names):
            kp_dict = dict()
            kp_dict["Name"] = key
            kp_dict["Part ID"] = str(int(idx) + 1)
            kp_dict["scale"] = np.array([1.0, 1.0, 1.0]).tolist()
            kp_dict["location_wrt_cam"] = kps_9_cam_np[idx].tolist()
            if key == "Link0":
                kp_dict["R2C_mat"] = this_rot.detach().cpu().numpy()[0]
            meta_dict["keypoints"].append(kp_dict)
        
        for idx, key in enumerate(joint_7_names):
            joint_dict = {}
            joint_dict["Name"] = key
            joint_dict["position"] = this_joint_angles_np[idx][0]
            meta_dict["joints"].append(joint_dict)
        
        for idx, key in enumerate(joint_14_names):
            joint_dict = {}
            joint_dict["Name"] = key
            joint_dict["location_wrt_cam"] = kps_14_cam_np[idx].tolist()
            meta_dict["joints_3n_fixed_42"].append(joint_dict)
        
        meta_json.append(meta_dict)

        
        file_write_meta = open(path_meta, 'wb')
        pickle.dump(meta_json, file_write_meta)
        file_write_meta.close()
        
        
        





















