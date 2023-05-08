import torch
import numpy as np
import math
import time
import kaolin as kal
import random
import torch.nn.functional as F
import cv2
import os
from copy import deepcopy

def projectiveprojection_real(cam, x0, y0, w, h, nc=0.01, fc=10.0):
    # this is for center view
    # NOTE: only return a 3x1 vector (diagonal??)
    q = -(fc + nc) / float(fc - nc)
    qn = -2 * (fc * nc) / float(fc - nc)
    fx = cam[0, 0]
    fy = cam[1, 1]
    px = cam[0, 2]
    py = cam[1, 2]
    """
    # transpose: compensate for the flipped image
    proj_T = [
            [2*fx/w,          0,                0,  0],
            [0,               2*fy/h,           0,  0],
            [(-2*px+w+2*x0)/w, (2*py-h+2*y0)/h, q,  -1],
            [0,               0,                qn, 0],
        ]
        sometimes: P[1,:] *= -1, P[2,:] *= -1
        # Third column is standard glPerspective and sets near and far planes
    """
    # Draw our images upside down, so that all the pixel-based coordinate systems are the same
    if isinstance(cam, np.ndarray):
        proj_T = np.zeros((4, 4), dtype=np.float32)
    elif isinstance(cam, torch.Tensor):
        proj_T = torch.zeros(4, 4).to(cam)
    else:
        raise TypeError("cam should be ndarray or tensor, got {}".format(type(cam)))
    proj_T[0, 0] = 2 * fx / w
    proj_T[1, 0] = -2 * cam[0, 1] / w  # =0
    proj_T[1, 1] = 2 * fy / h
    proj_T[2, 0] = (-2 * px + w + 2 * x0) / w
    proj_T[2, 1] = (+2 * py - h + 2 * y0) / h
    proj_T[2, 2] = q
    proj_T[3, 2] = qn
    proj_T[2, 3] = -1.0
    return proj_T

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

def load_part_mesh(paths, device):
    # Return lists of [vertices1, vertices2, ...]
    # [faces1, faces2, ...]
    vertices_list, faces_list, normals_list, faces_num_list = [], [], [], []
    pre_num_vertices = 0
    pre_num_faces = 0
    for idx, path in enumerate(paths):
        this_mesh = kal.io.obj.import_mesh(path, with_materials=True, with_normals=True)
        vertices = this_mesh.vertices.unsqueeze(0).to(device)
        normals = this_mesh.vertex_normals.to(device).unsqueeze(0)
        vertices.requires_grad = False
        faces = this_mesh.faces.to(device)
        faces.requires_grad=False
        faces += pre_num_vertices
        pre_num_vertices += vertices.shape[1]
        pre_num_faces = faces.shape[0]
        #print("faces.shape", pre_num_faces)
        
        vertices_list.append(vertices)   
        faces_list.append(faces)
        normals_list.append(normals)
        faces_num_list += [idx+1 for j in range(9 * pre_num_faces)]
    
    return vertices_list, faces_list, normals_list, faces_num_list
    
def load_binary_part_mesh(paths, device):
    # Return lists of [vertices1, vertices2, ...]
    # [faces1, faces2, ...]
    vertices_list, faces_list, normals_list, faces_num_list = [], [], [], []
    pre_num_vertices = 0
    pre_num_faces = 0
    for idx, path in enumerate(paths):
        this_mesh = kal.io.obj.import_mesh(path, with_materials=True, with_normals=True)
        vertices = this_mesh.vertices.unsqueeze(0).to(device)
        normals = this_mesh.vertex_normals.to(device).unsqueeze(0)
        vertices.requires_grad = False
        faces = this_mesh.faces.to(device)
        faces.requires_grad=False
        faces += pre_num_vertices
        pre_num_vertices += vertices.shape[1]
        pre_num_faces = faces.shape[0]
        #print("faces.shape", pre_num_faces)
        
        vertices_list.append(vertices)   
        faces_list.append(faces)
        normals_list.append(normals)
        faces_num_list += [1 for j in range(9 * pre_num_faces)]
    
    return vertices_list, faces_list, normals_list, faces_num_list

#def concat_part_mesh(vertices_list, trans_list, angles_list, device, joints_x3d=False):
#    ori_trans_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.333], [0.0,0.0,0.0],\
#                                        [0.0,-0.316,0.0], [0.0825,0.0,0.0],[-0.0825,0.384,0.0],\
#                                        [0.0,0.0,0.0], [0.088,0.0,0.0],[0.0,0.0,0.107],\
#                                        [0.0,0.0,0.0584], [0.0,0.0,0.0]
#                                        ])).float().to(device)
#    ori_angles_list = torch.from_numpy(np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
#              [1.5707963267948966,0.0,0.0], [1.5707963267948966,0.0,0.0], [-1.5707963267948966,0.0,0.0],\
#               [1.5707963267948966,0.0,0.0],  [1.5707963267948966,0.0,0.0], [0.0,-1.5707963267948966/2,0.0],\
#               [0.0,0.0,0.0], [0.0,0.0,0.0]
#                                            ])).float().to(device)  
#                                            
#    joints_info = [
#              {"base" : 0, "offset" : [0.0, 0.0, 0.14]},
#              {"base" : 1, "offset" : [0.0, 0.0, 0.0]},
#              {"base" : 3, "offset" : [0.0, 0.0, -0.1210]},
#              {"base" : 4, "offset" : [0.0, 0.0, 0.0]},
#              {"base" : 5, "offset" : [0.0, 0.0, -0.2590]},
#              {"base" :5, "offset" : [0.0, 0.0158, 0.0]},
#              {"base" : 7, "offset" : [0.0, 0.0, 0.0520]},
#              {"base" : 8, "offset" : [0.0, 0.0, 0.0584]}
#              ]
#
#    length = len(vertices_list)
##    assert length == len(ori_angles_list)
##    assert length == len(ori_trans_list)
#    #print("angles_list.shape", angles_list.shape)
#    #print("trans_list.shape", trans_list.shape)
#    assert len(angles_list[0]) == 7
#    assert len(trans_list[0]) == 1
#    #print("vertcies_list", vertices_list)
#    vertices_list_ = []
#    ori_mat = torch.eye(3).to(device)
#    ori_trans = torch.zeros(3).to(device)
#    kps_list, R2C_list, joints_x3d_rob = [], [], []
#    for idx, sample in enumerate(zip(vertices_list, ori_trans_list[:length], ori_angles_list[:length])):
#        # print(idx)
#        vertices, trans, angles = sample
#        if idx == 0:
#            vertices_list_.append(vertices)
#            kps_list.append(ori_trans.clone())
#            R2C_list.append(ori_mat.clone())
#            continue
#        
#        this_mat = euler_angles_to_matrix(angles,convention="XZY").squeeze(0).T.to(device)
#        if 1<= idx <= 7:
#            new_mat = _axis_angle_rotation("Z", angles_list[0][idx-1]).reshape(3,-1).T
#            this_mat = new_mat @ this_mat
#
#        if idx == 9:
#            trans = trans + trans_list[0] * torch.tensor([0.0, 1.0, 0.0]).to(device)
#        if idx == 10:
#            trans = trans + (-2) * trans_list[0] * torch.tensor([0.0, 1.0, 0.0]).to(device)
#                
#        ori_trans += trans @ ori_mat
#        ori_mat = this_mat @ ori_mat
#        new_vertices = vertices @ ori_mat + ori_trans
#        vertices_list_.append(new_vertices)
#        kps_list.append(ori_trans.clone())
#        R2C_list.append(ori_mat.clone())
#
#        
#        
#    if joints_x3d:
#        for idx in range(len(joints_info)):
#            base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"])).unsqueeze(0).to(device)
#            joints_x3d_rob.append(offset.float() @ R2C_list[base_idx].float() + kps_list[base_idx].float())
#        
#    #print("vertices_list", vertices_list)
##    print("kps_list", kps_list)
##    print("joints_x3d_rob", joints_x3d_rob)
#    if joints_x3d:
#        return vertices_list_, torch.cat(joints_x3d_rob, dim=0), R2C_list, kps_list
#    return vertices_list_, R2C_list, kps_list

def concat_part_mesh(vertices_list, Angles, device, joints_x3d=False):
    # Angles.shape : B x N x 1
    B, N, _ = Angles.shape
    #assert N == 8

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

    length = len(vertices_list)
    
    vertices_list_ = []
    ori_mat = torch.eye(3).repeat(B,1).reshape(B,3,3).float().to(device)
    ori_trans = torch.zeros(B, 3, 1).float().to(device)
    ori_angles = euler_angles_to_matrix(ori_angles_list, convention="XYZ")
    
    kps_list, R2C_list, joints_x3d_rob = [], [], []
    for j in range(ori_trans_list.shape[0]):
        # print(idx)
        this_vertices = vertices_list[j].repeat(B, 1, 1)
        #print("vertices.shape",this_vertices.shape) # 1 x num_pts x 3
        if j == 0:
            vertices_list_.append(this_vertices)
            kps_list.append(ori_trans.clone())
            R2C_list.append(ori_mat.clone())
            continue
        
        this_mat = ori_angles[j].repeat(B, 1).reshape(B, 3, 3)
        trans = ori_trans_list[j].unsqueeze(0).repeat(B, 1)
        if 1<= j <= 7:
            new_mat = _axis_angle_rotation("Z", Angles[:, j-1, 0])
            this_mat = torch.bmm(this_mat, new_mat)

#        if j == 9:
#            trans = trans + Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
#        if j == 10:
#            trans = trans + (-2) * Angles[:, -1:,0] * (torch.tensor([[0.0, 1.0, 0.0]]).to(device)).repeat(B,1)
                
        ori_trans += torch.bmm(ori_mat, trans.unsqueeze(2))
        ori_mat = torch.bmm(ori_mat, this_mat)
        #new_vertices = vertices @ ori_mat + ori_trans
        new_vertices = (torch.bmm(ori_mat, this_vertices.permute(0, 2, 1)) + ori_trans).permute(0, 2, 1)
        vertices_list_.append(new_vertices)
        kps_list.append(ori_trans.clone())
        R2C_list.append(ori_mat.clone())

        
    joints_x3d_rob = torch.zeros(B, N, 3, 1).to(device)   
    if joints_x3d:
        for idx in range(len(joints_info)):
            base_idx, offset = joints_info[idx]["base"], torch.from_numpy(np.array(joints_info[idx]["offset"]))[None, :, None].repeat(B, 1, 1).to(device)
            this_x3d = torch.bmm(R2C_list[base_idx], offset.float()) + kps_list[base_idx]
            joints_x3d_rob[:, idx, :, :] = this_x3d.clone()
        joints_x3d_rob = joints_x3d_rob.squeeze(3)
        
        
    #print("vertices_list", vertices_list)
#    print("kps_list", kps_list)
#    print("joints_x3d_rob", joints_x3d_rob)
    if joints_x3d:
        return vertices_list_, joints_x3d_rob, R2C_list, kps_list
    return vertices_list_, R2C_list, kps_list

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

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
    
def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])
    
def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

#def seg_and_transform(im_features, mask, Rot_matrix,trans_matrix, device, rot_list=[], trans_list=[]):
#    # rot_list ´æpart to camera pose
#    im_features_clone = im_features.clone()
#    pre_faces = 0.0
#    im_features_list, mask_list = [], []
#    #print("trans_list", trans_list)
#    for idx in range(len(rot_list)):
#        mask_index = torch.where(torch.abs(mask-idx-1)<1e-3)
#        this_im_feature = im_features[mask_index]
#        if mask_index[0].detach().cpu().numpy().tolist() != []:
#            if rot_list != [] and trans_list == []:
#                this_im_feature_ = (this_im_feature.reshape(-1,3) @ rot_list[idx]) @ Rot_matrix
#            elif rot_list != [] and trans_list != []: 
#                print("this_im_feature.shape", this_im_feature.shape)
#                this_im_feature_ = (this_im_feature.reshape(-1,3) @ rot_list[idx] + trans_list[idx]) @ Rot_matrix + trans_matrix
#            else:
#                raise ValueError
#            im_features_clone[mask_index] = this_im_feature_.reshape(-1)
#        im_features_list.append(this_im_feature.clone())
#    
#    return im_features_list, im_features_clone
    
def seg_and_transform(im_features, mask, Rot_matrix,basis_change_matrix, trans_matrix, device, rot_list=[], trans_list=[]):
    # rot_list ´æpart to camera pose
    im_features_clone = im_features.clone()
    pre_faces = 0.0
    im_features_list, mask_list = [], []
    #print("trans_list", trans_list)
#    print("im_features.shape", im_features.shape)
#    print("mask.shape", mask.shape)
    for j in range(im_features.shape[0]):
        this_mask = mask[j]
        this_im_features = im_features[j]
        for idx in range(len(rot_list)):
            mask_index = torch.where(torch.abs(this_mask-idx-1)<1e-3)
            this_im_feature = this_im_features[mask_index]
            if mask_index[0].detach().cpu().numpy().tolist() != []:
                if rot_list != [] and trans_list != []: 
                    this_im_feature_ = ((this_im_feature.reshape(-1,3) @ rot_list[idx][j].T + trans_list[idx][j].T) @ Rot_matrix[j].T + trans_matrix[j][None, :]) @ basis_change_matrix[j] 
                else:
                    raise ValueError
                im_features_clone[j][mask_index] = this_im_feature_.reshape(-1)
            
    
    return im_features_clone

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

#def exists_or_mkdir(path):
#    if not os.path.exists(path):
#        os.makedirs(path)
#        return False
#    else:
#        return True

def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    assert poses.shape[-1] == 6
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    matrix = torch.stack((x, y, z), -1)
    return matrix

def pixel2world(x, y, z, img_width, img_height, fx, fy, cx=None, cy=None):
    """Converts image coordinates to 3D real world coordinates using depth values

    Parameters
    ----------
    x: np.array
        Array of X image coordinates.
    y: np.array
        Array of Y image coordinates.
    z: np.array
        Array of depth values for the whole image.
    img_width: int
        Width image dimension.
    img_height: int
        Height image dimension.
    fx: float
        Focal of the camera over X axis.
    fy: float
        Focal of the camera over Y axis.
    cx: float
        X coordinate of principal point of the camera.
    cy: float
        Y coordinate of principal point of the camera.

    Returns
    -------
    w_x: np.array
        Array of X world coordinates.
    w_y: np.array
        Array of Y world coordinates.
    w_z: np.array
        Array of Z world coordinates.

    """
    if cx is None:
        cx = img_width / 2
    if cy is None:
        cy = img_height / 2
    w_x = (x - cx) * z / fx
    w_y = (cy - y) * z / fy
    w_z = z
    return w_x, w_y, w_z
    
def depthmap2points(image, fx, fy, cx=None, cy=None):
    """Converts image coordinates to 3D real world coordinates using depth values

    Parameters
    ----------
    image: np.array
        Array of depth values for the whole image.
    fx: float
        Focal of the camera over X axis.
    fy: float
        Focal of the camera over Y axis.
    cx: float
        X coordinate of principal point of the camera.
    cy: float
        Y coordinate of principal point of the camera.

    Returns
    -------
    points: np.array
        Array of XYZ world coordinates.

    """
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h)) 
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, fx, fy, cx, cy)
    return points


def depthmap2pointcloud(depth, fx, fy, cx=None, cy=None):
    points = depthmap2points(depth, fx, fy, cx, cy)
    points = points.reshape((-1, 3))
    return points














