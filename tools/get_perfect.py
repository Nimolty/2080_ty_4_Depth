import os
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

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
    
def depth_to_xyz(depthImage, f, scale_h=1., scale_w=1.):
    # input depth image[B, 1, H, W]
    # output xyz image[B, 3, H, W]

    fx = f * scale_w
    fy = f * scale_h
    B, C, H, W = depthImage.shape
    device = depthImage.device
    du = W//2 - 0.5
    dv = H//2 - 0.5

    xyz = torch.zeros([B, H, W, 3], device=device)
    imageIndexX = torch.arange(0, W, 1, device=device) - du
    imageIndexY = torch.arange(0, H, 1, device=device) - dv
    depthImage = depthImage.squeeze()
    if B == 1:
        depthImage = depthImage.unsqueeze(0)

    xyz[:, :, :, 0] = depthImage/fx * imageIndexX
    xyz[:, :, :, 1] = (depthImage.transpose(1, 2)/fy * imageIndexY.T).transpose(1, 2)
    xyz[:, :, :, 2] = depthImage
    xyz = xyz.permute(0, 3, 1, 2).to(device)
    return xyz

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


def get_surface_normal(x, f, scale_h, scale_w):
    xyz = depth_to_xyz(x, f, scale_h, scale_w)
    dx,dy = gradient(xyz)
    surface_normal = torch.cross(dx, dy, dim=1)
    surface_normal = surface_normal / (torch.norm(surface_normal,dim=1,keepdim=True)+1e-8)
    return surface_normal, dx, dy

if __name__ == "__main__":
    root_dir = "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/04751/0010_simDepthImage.exr"
    simdepth_file = cv2.imread(root_dir, cv2.IMREAD_UNCHANGED)
    simdepth = simdepth_file[:, :, 0][:, :, None]
    pcs = DepthToPointCloud(simdepth, f=502.30)
    
    simdepth_tensor = torch.from_numpy(simdepth_file[:, :, 0][None, None, :, : ]) #Bx1XHXW
    surface_normal, _, _ = get_surface_normal(simdepth_tensor, f=502.30, scale_h=1.0, scale_w=1.0)
    
    surface_normal = surface_normal.squeeze().permute(1,2,0).cpu().numpy()
    
    all_pts = np.concatenate([pcs, surface_normal], axis=-1)
    all_pts = all_pts[:, 140:500, :]
    all_pts = cv2.resize(all_pts, (400,400), interpolation=cv2.INTER_NEAREST)
    
    print(pcs.shape)
    print(surface_normal.shape)
    print(all_pts.shape)
    all_pts = all_pts.reshape(-1,6)
    np.savetxt(f"./raw_xyzn.txt", all_pts)
    