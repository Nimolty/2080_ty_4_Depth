#coding=gbk
# import imath
import numpy as np
import cv2
from PIL import Image as PILImage
from copy import deepcopy
import array
import os
import open3d as o3d
from tqdm import tqdm
import torch.nn.functional as F
import torch
import time
# import OpenEXR

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def shrink_and_crop_resolution(image_input_resolution, image_ref_resolution):
 
    image_input_width, image_input_height = image_input_resolution
    image_ref_width, image_ref_height = image_ref_resolution

    # Casting to float for Py2
    scale_factor_based_on_width = float(image_input_width) / float(image_ref_width)
    image_ref_height_based_on_width = int(
        scale_factor_based_on_width * image_ref_height
    )

    # Casting to float for Py2
    scale_factor_based_on_height = float(image_input_height) / float(image_ref_height)
    image_ref_width_based_on_height = int(
        scale_factor_based_on_height * image_ref_width
    )

    if image_input_width >= image_ref_width_based_on_height:
        image_input_cropped_resolution = (
            image_ref_width_based_on_height,
            image_input_height,
        )
    else:
        assert image_input_height >= image_ref_height_based_on_width
        image_input_cropped_resolution = (
            image_input_width,
            image_ref_height_based_on_width,
        )
 
    image_input_cropped_coords = (
        (image_input_width - image_input_cropped_resolution[0]) // 2,
        (image_input_height - image_input_cropped_resolution[1]) // 2,
    )

    return image_input_cropped_resolution, image_input_cropped_coords

# 我们统一规定一下读进来的图片均转成RGB，这样的话cv2和PILImage包都可以使用
def crop_and_resize_img(image_path, input_resolution):
    img_start = time.time()
    img = cv2.imread(image_path)
    img_read = time.time()
    #print("img_read", img_read - img_start)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 转成了RGB了
    #print("img_trans", time.time() - img_read)
    img_ori_h, img_ori_w, _ = img.shape
    input_h, input_w = input_resolution # 要求是h x w的形状
    before = time.time()
    img_input_cropped_resolution, img_input_cropped_coords = shrink_and_crop_resolution([img_ori_w, img_ori_h], [input_w, input_h])
    img_middle = time.time()
    crop_w, crop_h = img_input_cropped_coords
    img = img[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC) # 双三次插值
    img_final = time.time()
    #print("img_before", before-  img_start)
    #print("img_middle", img_middle - before)
    #print("img_final", img_final - img_middle)
#    print('max', np.max(img))
#    print('min', np.min(img))
    
    return img

# 深度图转为点云
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

def res_crop_and_resize_simdepth(simdepth_path, input_resolution, uv=False, xy=False, nrm=False, device="cpu", \
                              camera_K=None, f=502.30,threshold_dist=[0.01, 7.00], threshold_vs=[-0.1, 7.00]):
    start_time = time.time()              
    input_h, input_w = input_resolution
    simdepth_file = cv2.imread(simdepth_path, cv2.IMREAD_UNCHANGED)
    img_ori_h, img_ori_w, _ = simdepth_file.shape
    th_lw, th_up = threshold_dist
    th_lwv, th_upv = threshold_vs
    
    simdepth = simdepth_file[:, :, 0][:, :, None]
    simdepth[np.where(simdepth <= th_lw)] = th_lwv
    simdepth[np.where(simdepth >= th_up)] = th_upv
    
    
    img_input_cropped_resolution, img_input_cropped_coords = shrink_and_crop_resolution([img_ori_w, img_ori_h], [input_w, input_h])
    crop_w, crop_h = img_input_cropped_coords
    
    simdepth_crop = simdepth[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
    simdepth_crop = cv2.resize(simdepth_crop, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :] # 最近邻 1 x h x w
    
    # 
    simdepth_rp = np.mean(simdepth_crop) + 1e-7
    
    # 
    simdepth_crop_res = simdepth_crop - simdepth_rp
    
    crop_time = time.time()
    #print("crop_time", crop_time - start_time)
    
    if xy:
        pc = DepthToPointCloud(simdepth, f)
        x_wrt_cam, y_wrt_cam = pc[:, :, 0][:, :, None], pc[:, :, 1][:, :, None]
        x_wrt_cam = x_wrt_cam[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        y_wrt_cam = y_wrt_cam[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        x_wrt_cam = cv2.resize(x_wrt_cam, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :] # 1 x h x w
        y_wrt_cam = cv2.resize(y_wrt_cam, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :]
        x_rp, y_rp = np.mean(x_wrt_cam), np.mean(y_wrt_cam)
        
        # 
        x_wrt_cam_res = x_wrt_cam / (simdepth_crop_res + 1e-7) - x_rp / simdepth_rp
        y_wrt_cam_res = y_wrt_cam / (simdepth_crop_res + 1e-7) - y_rp / simdepth_rp
        xy_wrt_cam_res = np.concatenate([x_wrt_cam_res, y_wrt_cam_res],axis=0)
    else:
        xy_wrt_cam_res, x_rp, y_rp = None, None, None
    
    xy_time = time.time()
    #print("xy_time", xy_time - crop_time)
    # Expect camera_K is numpy
    u_rp, v_rp, _ = camera_K @ np.array([x_rp / simdepth_rp, y_rp / simdepth_rp, 1])
    
    if nrm:
        assert xy_wrt_cam_res is not None

        simdepth_tensor = torch.from_numpy(simdepth_file[:, :, 0][None, None, :, : ]).to(device) #Bx1XHXW
        surface_normal, _, _ = get_surface_normal(simdepth_tensor, f, scale_h=1.0, scale_w=1.0)
        normals = surface_normal.cpu().numpy()[0].transpose(1, 2, 0)
        normals_crop = normals[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        normals_crop = cv2.resize(normals_crop, (input_w, input_h), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    else:
        normals_crop = None
    
    nrm_time = time.time()
    #print("nrm_time", nrm_time - xy_time)
    
    if uv:
        u = np.arange(0, img_ori_w)[None, :].repeat(img_ori_h, axis=0)
        v = np.arange(0, img_ori_h)[:, None].repeat(img_ori_w, axis=1)
        u = u[:, :, None]
        v = v[:, :, None]
        u = u[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        v = v[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        u = cv2.resize(u, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :]
        v = cv2.resize(v, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :]
        
        # Residual 
        u_res, v_res = u - u_rp, v - v_rp
        uv_res = np.concatenate([u_res, v_res], axis=0)
    else:
        uv_res = None
    
    uv_time = time.time()
    xyz_rp = np.array([x_rp, y_rp, simdepth_rp], dtype=np.float32)[None, :] # 1 x 3
    #print("uv_time", uv_time- nrm_time)
        
    return simdepth_crop_res, xy_wrt_cam_res, uv_res, normals_crop, xyz_rp

def crop_and_resize_simdepth(simdepth_path, input_resolution, uv=False, xy=False, nrm=False, device="cpu", \
                              f=502.30,threshold_dist=[0.01, 7.00], threshold_vs=[-0.1, 7.00]):
    start_time = time.time()              
    input_h, input_w = input_resolution
    simdepth_file = cv2.imread(simdepth_path, cv2.IMREAD_UNCHANGED)
    img_ori_h, img_ori_w, _ = simdepth_file.shape
    th_lw, th_up = threshold_dist
    th_lwv, th_upv = threshold_vs
    
    simdepth = simdepth_file[:, :, 0][:, :, None]
    simdepth[np.where(simdepth <= th_lw)] = th_lwv
    simdepth[np.where(simdepth >= th_up)] = th_upv
    
    img_input_cropped_resolution, img_input_cropped_coords = shrink_and_crop_resolution([img_ori_w, img_ori_h], [input_w, input_h])
    crop_w, crop_h = img_input_cropped_coords
    
    simdepth_crop = simdepth[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
    simdepth_crop = cv2.resize(simdepth_crop, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :] # 最近邻 1 x h x w
    
    crop_time = time.time()
    #print("crop_time", crop_time - start_time)
    
    if xy:
        pc = DepthToPointCloud(simdepth, f)
        x_wrt_cam, y_wrt_cam = pc[:, :, 0][:, :, None], pc[:, :, 1][:, :, None]
        x_wrt_cam = x_wrt_cam[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        y_wrt_cam = y_wrt_cam[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        x_wrt_cam = cv2.resize(x_wrt_cam, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :]
        y_wrt_cam = cv2.resize(y_wrt_cam, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :]
        xy_wrt_cam = np.concatenate([x_wrt_cam, y_wrt_cam],axis=0)
    else:
        xy_wrt_cam = None
    
    xy_time = time.time()
    #print("xy_time", xy_time - crop_time)
    
    if nrm:
        assert xy_wrt_cam is not None

        simdepth_tensor = torch.from_numpy(simdepth_file[:, :, 0][None, None, :, : ]).to(device) #Bx1XHXW
        surface_normal, _, _ = get_surface_normal(simdepth_tensor, f, scale_h=1.0, scale_w=1.0)
        normals = surface_normal.cpu().numpy()[0].transpose(1, 2, 0)
        normals_crop = normals[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        normals_crop = cv2.resize(normals_crop, (input_w, input_h), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    else:
        normals_crop = None
    
    nrm_time = time.time()
    #print("nrm_time", nrm_time - xy_time)
    
    if uv:
        u = np.arange(0, img_ori_w)[None, :].repeat(img_ori_h, axis=0)
        v = np.arange(0, img_ori_h)[:, None].repeat(img_ori_w, axis=1)
        u = u[:, :, None]
        v = v[:, :, None]
        u = u[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        v = v[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w, :]
        u = cv2.resize(u, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :]
        v = cv2.resize(v, (input_w, input_h), interpolation=cv2.INTER_NEAREST)[None, :, :]
        uv = np.concatenate([u, v], axis=0)
    else:
        uv = None
    
    uv_time = time.time()
    #print("uv_time", uv_time- nrm_time)
        
    return simdepth_crop, xy_wrt_cam, uv, normals_crop


def crop_and_resize_mask(mask_path, input_resolution, mask_dict, num_classes):
    # Expect num_classes 为 3 或者 len(mask_dict) + 1
    start_time = time.time()
    input_h, input_w = input_resolution
    mask_file = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    red = mask_file[:, :, 2] # 因为这里是bgr
    red_height, red_width = red.shape[0], red.shape[1]
     
    
    pre_time = time.time()
    
    if num_classes == len(mask_dict) + 1:
        # 这里就是生成所有part的mask
        mask_res = np.zeros((red_height, red_width))
        for idx, (key, value) in enumerate(mask_dict.items()):
            mask_res[np.where(red == value)] = idx+1
    elif num_classes == 3:
#        mask_res = np.zeros((red_height, red_width))
#        for idx, (key, value) in enumerate(mask_dict.items()):
#            if key == "Link0":
#                mask_res[np.where(red == value)] = 1
#            else:
#                mask_res[np.where(red == value)] = 2
        mask_res = np.ones((red_height, red_width)) * 2
        value_0 = mask_dict["Link0"]
        mask_res[np.where(red == value_0)] = 1
        mask_res[np.where(red == 1.0)] = 0
    else:
        raise ValueError
    middle_time = time.time()
    
    img_ori_h, img_ori_w, _ = mask_file.shape
    img_input_cropped_resolution, img_input_cropped_coords = shrink_and_crop_resolution([img_ori_w, img_ori_h], [input_w, input_h])
    crop_w, crop_h = img_input_cropped_coords
    mask_res = mask_res[crop_h:img_ori_h-crop_h, crop_w:img_ori_w-crop_w][:, :, None]
    mask_res = cv2.resize(mask_res, (input_w, input_h), interpolation=cv2.INTER_NEAREST) # 最近邻
    
    crop_time = time.time()
#    print("mask_pre", pre_time - start_time)
#    print("mask_middle", middle_time - pre_time)
#    print("mask_crop", crop_time - middle_time)
#    print("mask_res", mask_res.shape)
#    for class_id in range(mask_res.shape[2]):
#        image = PILImage.fromarray(np.uint8(mask_res[:, :, class_id]*255))
#        image.save(f"./{class_id}_check.png")
#   expect mask_res为HXw
    return mask_res

def normalize_image(inp, mean, std):
    # image : hxwxc, 0-255
    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - mean) /std
    inp = inp.transpose(2, 0, 1) # c x h x w
    return inp

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


def check_scaled_depth(depth, x, y, u, v):
    f = 502.30
    cx = 640//2 - 0.5
    cy = 360//2 - 0.5
    
    inverse_x = (u - cx) / f * depth
    inverse_y = (v - cy) / f * depth
#    print('inverse_x', inverse_x)
#    print('inverse_y', inverse_y)
#    print('x', x)
#    print('y', y)
    print(inverse_x.all() == x.all())
    print(inverse_y.all() == y.all())

def random_crop(img_size, expect_size):
    H, W = img_size
    tH, tW = expect_size
    
    margin_h = max(H - tH, 0)
    margin_w = max(W - tW, 0)
    y1 = random.randint(0, margin_h+1)
    x1 = random.randint(0, margin_w+1)
    return y1, x1

if __name__ == "__main__":
    image_path = "/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/03206/0029_color.png"
    mask_path = "/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/03206/0029_mask.exr"
    simdepth_path = "/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/03206/0029_simDepthImage.exr"
    test_path = "/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/03206"
    input_resolution = [400, 400]
    img = crop_and_resize_img(image_path, input_resolution)
    
    mask_dict =  {
                "Link0": 0.007843138,
                "Link1": 0.011764706,
                "Link2": 0.015686275,
                "Link3": 0.019607844,
                "Link4": 0.023529412,
                "Link5": 0.02745098,
                "Link6": 0.03137255,
                "Link7": 0.03529412,
                "Panda_hand" : 0.003921568859368563
                }
    mask = crop_and_resize_mask(mask_path, input_resolution, mask_dict, num_classes=2)
    
    print('img.shape', img.shape)
    print('mask.shape', mask.shape)
    
    frame_name = 0
    for i in tqdm(range(1)):
        frame_id = str(i).zfill(4)
        simdepth_path = os.path.join(test_path, frame_id + "_simDepthImage.exr")
        depth, x, y, u, v, normalsm = crop_and_resize_simdepth(simdepth_path, input_resolution, uv=True, xy=True, nrm=True)
        print('depth.shape', depth.shape)
        print('x.shape', x.shape)
        print('y.shape', y.shape)
        print('u.shape', u.shape)
        print("v.shape", v.shape)
        print("normalsm.shape", normalsm.shape)
#        if i == 0:
#            np.savetxt(f"./pcs.txt", normalsm.reshape(-1, 6))
        # print('normals', normals)
        # check_scaled_depth(depth, x, y, u, v)
    

    
    
    
    
    
    
    
    
    
    
    
    


