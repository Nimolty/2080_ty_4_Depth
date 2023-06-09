import json
from collections import defaultdict
from pathlib import Path

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torch
import time
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import copy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyquaternion import Quaternion
import glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" 

from depth_c2rp.utils.spdh_utils import augment_3d, depthmap2pointcloud, depthmap2points, overlay_points_on_image, mosaic_images, pointcloud2depthmap, hole_filling
from depth_c2rp.utils.spdh_utils import apply_depth_normalization_16bit_image, heatmap_from_kpoints_array, gkern, compute_rigid_transform

JOINT_NAMES = [f"panda_joint_3n_{i+1}" for i in range(14)]


def init_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


class Voxel_dataset(Dataset):
    def __init__(self, train_dataset_dir: Path, val_dataset_dir : Path, real_dataset_dir : Path, joint_names: list, run: list, init_mode: str = 'train', img_type: str = 'D',
                 raw_img_size: tuple = (640, 360), input_img_size: tuple = (384, 216), sigma: int = 4., norm_type: str = 'mean_std',
                 network_input: str = 'D', network_output: str = 'H', network_task: str = '3d_RPE',
                 depth_range: tuple = (500, 3380, 15), depth_range_type: str = 'normal', aug_type: str = '3d',
                 aug_mode: bool = True, noise: bool = False, demo: bool = False, load_mask: bool = False, mask_dict : dict = {}, unnorm_depth: bool=False, cx_delta: int = 0, cy_delta: int = 0, change_intrinsic: bool=False, uv_input : bool = False):
        assert init_mode in ['train', 'test']

        self.train_dataset_dir = Path(train_dataset_dir)
        self.val_dataset_dir = Path(val_dataset_dir)
        self.real_dataset_dir = Path(real_dataset_dir)
        self.JOINT_NAMES = joint_names
        self.run = run
        self._mode = init_mode
        self.img_type = img_type
        self.raw_img_size = raw_img_size
        self.input_img_size = input_img_size
        self.sigma = sigma
        self.norm_type = norm_type
        self.network_input = network_input
        self.network_output = network_output
        self.depth_range = depth_range
        self.depth_range_type = depth_range_type
        self.network_task = network_task
        self.aug_type = aug_type
        self._aug_mode = aug_mode
        self.aug_mode = aug_mode
        self.noise = noise
        self.demo = demo
        self.load_mask = load_mask
        self.mask_dict = mask_dict
        self.unnorm_depth = unnorm_depth
        self.change_intrinsic = change_intrinsic
        self.uv_input = uv_input
        self.data = self.load_data()

    def __len__(self):
        return len(self.data[self.mode])

    def __getitem__(self, idx):    
        t1 = time.time()
        sample = self.data[self.mode][idx].copy()
        # image loading (depth and/or RGB)
        if self.mode == "train" or self.mode == "val":
            #depth_img = cv2.imread(sample['depth_file'], cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32) * 1000 # milimeters 
            depth_img = np.load(sample["depth_file"]).astype(np.float32) * 1000
        elif self.mode == "real":
            #depth_img = cv2.imread(sample['depth_file'], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_img = np.load(sample["depth_file"]).astype(np.float32)
        else:
            raise ValueError
        t1_mid = time.time()
        depth_img = cv2.resize(depth_img, tuple(self.input_img_size), interpolation=cv2.INTER_NEAREST)
        
        H, W = depth_img.shape
        assert H == self.input_img_size[1]
        assert W == self.input_img_size[0]
        # RGB and depth scale
        scale_x, scale_y = depth_img.shape[1] / self.raw_img_size[0], depth_img.shape[0] / self.raw_img_size[1]
        # initialize intrinsic matrix
        if self.mode == "train" or self.mode == "val":
            cam_settings_data = json.load(open(str(self.train_dataset_dir / "_camera_settings.json"), 'r'))
        else:
            cam_settings_data = json.load(open(str(self.real_dataset_dir / "_camera_settings.json"), 'r'))
        fx, fy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fx"]* scale_x, cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fy"] * scale_y
        cx, cy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cx"] * scale_x, cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cy"] * scale_y 
            
        mask_file_res = None
        if self.load_mask:
            mask_file = cv2.imread(sample["mask_file"], cv2.IMREAD_UNCHANGED)[:, :, 2]
            mask_file_res = np.zeros_like(mask_file) 
            assert self.mask_dict != {}
            for idx, (key, value) in enumerate(self.mask_dict.items()):
                mask_file_res[np.where(mask_file == value)] = idx+1
            mask_file_res = cv2.resize(mask_file_res[:, :, None], (W, H), interpolation=cv2.INTER_NEAREST)
            
        
        
        # adapt depth image to "padding" depth range type
        if self.depth_range_type == 'padding':
            Z_min, Z_max, dZ = self.depth_range
            new_img_h = (Z_max - Z_min) // dZ
#            if self.mode == "real" or self.mode == "val":
#                depth_img = depth_img[12:-12, :]
            padding = int(np.abs(depth_img.shape[0] - new_img_h) // 2)
            #print("depth_img.shape", depth_img.shape)
            #print("torch.max", np.max(depth_img))
                
            depth_img = cv2.copyMakeBorder(depth_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)
        
        t2 = time.time()
        # copy unnormed depth img
        if self.unnorm_depth:
            depth_img_unnorm = depth_img.copy() / 1000

        
        if self.depth_range_type == "padding":
            assert W > H
            cx = cx
            cy = cy + 84.0
        else:
            cx = cx
            cy = cy 
        
        intrinsic = np.asarray([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        #print("intrinsic", intrinsic)
        
        
        joints_2D = np.zeros((sample['joints'].shape[0], 2))
        joints_3D_Z = np.zeros((sample['joints'].shape[0], 3)) 
        for n, joint in enumerate(sample["joints"]):
            point3d = np.asarray([joint[0], joint[1], joint[2]]) # right handed reference frame
            joints_3D_Z[n] = point3d.copy()
            u, v, w = (intrinsic @ point3d[:, None]).T.squeeze(0)
            u = u / w
            v = v / w
            joints_2D[n] = [u, v]
        joints_3D_Z_copy = joints_3D_Z.copy()
        #print("joints_3d_z", joints_3D_Z)
        #print("joints_2d", joints_2D)
        # load R2C Mat and Trans
        R2C_Mat_before_aug = np.array(sample["R2C_Mat"]) # 3 x 3
        R2C_Trans_before_aug = np.array(sample["R2C_Trans"]) # 3
        t3 = time.time()
        
        # apply 2d or 3d augmentation
        if self.aug_mode:
            if self.aug_type == '2d':
                if self.noise:
                    seq = iaa.Sequential([
                        iaa.Pepper(p=(0.1, 0.15)),
                        iaa.Affine(
                            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                            rotate=(-10, 10),
                        )
                    ]).to_deterministic()
                else:
                    seq = iaa.Sequential([
                        iaa.Affine(
                            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                            rotate=(-10, 10),
                        )
                    ]).to_deterministic()
                kps_tmp = ia.KeypointsOnImage([ia.Keypoint(x, y) for x, y in joints_2D], shape=depth_img.shape)
                depth_img, joints_2D = seq.augment(image=depth_img, keypoints=kps_tmp)
                joints_2D = joints_2D.to_xy_array()
                if "RGB" in self.img_type:
                    rgb_img = seq.augment(image=rgb_img)
            else:
                points = depthmap2pointcloud(depth_img / 1000, fx=fx, fy=fy, cx=intrinsic[0, 2],
                                             cy=intrinsic[1, 2])
                points = points[points[:, 2] != 0]
                depth_img, joints_3D_Z, R2C_Mat_after_aug, R2C_Trans_after_aug = augment_3d(intrinsic, points, depth_img, joints_3D_Z, R2C_Mat_before_aug, R2C_Trans_before_aug)
                depth_img = depth_img * 1000 
                if self.noise:
                    seq = iaa.Sequential([
                        iaa.Pepper(p=(0.1, 0.15)),
                    ]).to_deterministic()
                    depth_img = seq.augment(image=depth_img)
                point3d = joints_3D_Z.copy()
                joints_2D = (intrinsic @ point3d[..., None]).squeeze(-1)
                joints_2D = (joints_2D / joints_2D[:, -1:])[:, :2]
        else:
            R2C_Mat_after_aug = R2C_Mat_before_aug
            R2C_Trans_after_aug = R2C_Trans_before_aug
            
        if self.change_intrinsic and self.mode == "real":
            # initialize intrinsic matrix
            train_cam_settings_data = json.load(open(str(self.train_dataset_dir / "train_camera_settings.json"), 'r'))
            train_fx, train_fy = train_cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fx"]* scale_x, train_cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fy"]* scale_y
            # Pay Attention to the cy!!!
            train_cx, train_cy = train_cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cx"]* scale_x, train_cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cy"]* scale_y
            
            if self.depth_range_type == "padding":
                assert W > H
                train_cx = train_cx
                train_cy = train_cy + 84.0
                #cx = 191.7
                #cy = cx
                
            else:
                train_cx = train_cx
                train_cy = train_cy
            
            train_intrinsic = np.asarray([
            [train_fx, 0, train_cx],
            [0, train_fy, train_cy],
            [0, 0, 1],   
            ])
            
            points = depthmap2pointcloud(depth_img / 1000, fx=fx, fy=fy, cx=intrinsic[0, 2],
                                             cy=intrinsic[1, 2])
            points = points[points[:, 2] != 0]
            depth_img = pointcloud2depthmap(points, depth_img.shape[1], depth_img.shape[0],
                                          fx=train_intrinsic[0, 0],
                                          fy=train_intrinsic[1, 1],
                                          cx=train_intrinsic[0, 2],
                                          cy=train_intrinsic[1, 2]).astype(depth_img.dtype)
            depth_img = hole_filling(depth_img, kernel_size=2) * 1000
            point3d = joints_3D_Z.copy()
            #point3d[..., 1] *= -1  # I dont' known whether to invert Y axis direction for 2D reprojection
            joints_2D = (train_intrinsic @ point3d[..., None]).squeeze(-1)
            joints_2D = (joints_2D / joints_2D[:, -1:])[:, :2]
            
            intrinsic = train_intrinsic
            fx = train_fx
            fy = train_fy
            cx = train_cx
            cy = train_cy
        
            
        #print(intrinsic)
        # create visible depth map
        depth_img_vis = ((depth_img * 255) / depth_img.max()).astype(np.uint8)[..., None]
        depth_img_vis = np.concatenate([depth_img_vis, depth_img_vis, depth_img_vis], -1)
        
        R2C_Pose_after_aug = np.eye(4, 4)
        R2C_Pose_after_aug[:3, :3] = R2C_Mat_after_aug
        R2C_Pose_after_aug[:3, 3] = R2C_Trans_after_aug 
        t4 = time.time()

        # get 3D joints with Z value from depth map or from ground truth values
        z_values = depth_img.copy() / 1000  # depth values from millimeters to meters
        joints_2D_homo = np.ones((joints_2D.shape[0], 3))
        joints_2D_homo[:, :2] = joints_2D
        XY_rw = (np.linalg.inv(intrinsic) @ joints_2D_homo[..., None]).squeeze(-1)[:, :2]
        joints_3D_depth = np.ones((joints_2D.shape[0], 3), dtype=np.float32)
        joints_3D_depth[:, :2] = XY_rw
        depth_coords = []
        for joint in joints_2D:
            x, y = joint[0], joint[1]
            if x < 0:
                x = 0
            if x > z_values.shape[1] - 1:
                x = z_values.shape[1] - 1
            if y < 0:
                y = 0
            if y > z_values.shape[0] - 1:
                y = z_values.shape[0] - 1
            depth_coords.append([x, y])
        depth_coords = np.array(depth_coords)
        z = z_values[depth_coords[:, 1].astype(int), depth_coords[:, 0].astype(int)]
        joints_3D_depth *= z[..., None]
        
        t5 = time.time()
        # compute XYZ image
        if "XYZ" in self.network_input:
            xyz_img = depthmap2points(depth_img / 1000, fx=fx, fy=fy, cx=intrinsic[0, 2], cy=intrinsic[1, 2])
            xyz_img[depth_img == 0] = 0
            
            xyz_img_scale = xyz_img.copy()
            if self.mode == "real" or self.mode == "val":
                #depth_img = depth_img[12:-12, :]
                xyz_img_scale[84:96, :, :] = 0
                xyz_img_scale[288:300, :, :] = 0
            xyz_img_scale[..., 0] = xyz_img_scale[..., 0] / 3.
            xyz_img_scale[..., 1] = xyz_img_scale[..., 1] / 2.
            xyz_img_scale[..., 2] = xyz_img_scale[..., 2] / 5.

        # depth map and keypoints normalization
        depth_img = apply_depth_normalization_16bit_image(depth_img, self.norm_type)
        if self.network_task == '2d_RPE':
            joints_2D[:, 0] = joints_2D[:, 0] / depth_img.shape[1]
            joints_2D[:, 1] = joints_2D[:, 1] / depth_img.shape[0]
        else:
            Z_min, Z_max, dZ = self.depth_range
            sigma_mm = 50

            # UV heatmaps
            x, y = np.meshgrid(np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0]))
            heatmaps_uv = np.zeros((joints_2D.shape[0], depth_img.shape[0], depth_img.shape[1]))
            
            # set ind and offsets
            # ind : H x H x H district
            ind = np.zeros((joints_2D.shape[0], 1))
            offset = np.zeros((joints_2D.shape[0], 2))
            
            for n, (p, P) in enumerate(zip(joints_2D, joints_3D_Z)): 
                P1 = P.copy() * 1000
                # compute distances (px) from point                
                if p[0] < 0.0 or p[1] < 0.0 or p[0] >= depth_img.shape[1] or p[1] >= depth_img.shape[0]:
                    x_int, y_int, x_off, y_off = 0, 0, 0, 0
                    x_ind, y_ind = 0, 0
                else:  
                    x_int, y_int = np.floor(p[0]), np.floor(p[1])
                    x_off, y_off = p[0] - x_int, p[1] - y_int
                    x_ind, y_ind = x_int, y_int
                
                dst = np.sqrt((x - x_int) ** 2 + (y - y_int) ** 2)
                
                # set offset and ind
                offset[n][0] = x_off
                offset[n][1] = y_off
                
                ind[n][0] += (x_ind)    
                ind[n][0] += (y_ind) * depth_img.shape[1]   
                
                
                # convert sigma from mm to pixel
                sigma_pixel = sigma_mm * intrinsic[0, 0] / P1[2]
                # compute the heatmap
                mu = 0.
                heatmaps_uv[n] = np.exp(-((dst - mu) ** 2 / (2.0 * sigma_pixel ** 2)))
            
            heatmaps_25d = heatmaps_uv
        t6 = time.time()
        joints_2d_dz = ((joints_3D_Z[:, 2:3] * 1000 - Z_min) / dZ)
        #print("intrinsic", intrinsic)
        #print(ind)
        #print("t4  - t3", t4 - t3)
        
#        if "000001" in sample["depth_file"]: 
#            print("joints_3D_Z", joints_3D_Z)
#            print("joints_2D_depth", joints_2D)

        
        output = {
            'depth': torch.from_numpy(depth_img[None, ...]),
            'depthvis': depth_img_vis,
            'z_values': z_values,
            'K_depth': intrinsic.astype(np.float32), 
            'joints_2D_depth': joints_2D,
            "joints_2D_uv" : np.array(joints_2D),
            'joints_3D_Z': joints_3D_Z, 
            "joints_2d_dz" : joints_2d_dz,
            "joints_3D_kps" : torch.from_numpy(np.array(sample["joints_kps"]))[[0,2,3,4,6,7,8]],
            "rgb_path" : sample['rgb_file'],
            "joints_7" : torch.from_numpy(np.array(sample["joints_8"])[:7]).float(),
            "R2C_Pose" : R2C_Pose_after_aug[:3, :], #torch.from_numpy(R2C_Pose_after_aug)[:3, :].float(), 
            "depth_path" : sample['depth_file'],
            "uv_off": torch.from_numpy(offset),
            "uv_ind": torch.from_numpy(ind).type(torch.int64), 
        }
        if self.network_task == '2d_RPE':
            output['joints_3D_depth'] = joints_3D_depth
            output['heatmap_depth'] = torch.from_numpy(heatmaps.astype(np.float32).transpose(2, 0, 1))
            if "RGB" in self.network_input:
                output['heatmap_rgb'] = torch.from_numpy(heatmaps.astype(np.float32).transpose(2, 0, 1))
        else:
            output['heatmap_25d'] = torch.from_numpy(heatmaps_25d.astype(np.float32))
        if "XYZ" in self.network_input:
            output['xyz_img'] = xyz_img  # H x W x 3
            rgb_img = xyz_img.transpose(2, 0, 1) # 3 x H x W
            
            if self.uv_input:
                u_input, v_input = np.meshgrid(np.arange(xyz_img.shape[1]), np.arange(xyz_img.shape[0]))
                output["rgb_img"] = np.concatenate([rgb_img, u_input[None, ...], v_input[None, ...]], axis=0)
            else:
                output["rgb_img"] = rgb_img
            
            output["xyz_img_scale"] = xyz_img_scale.transpose(2, 0, 1)
        if self.load_mask:
            output["mask"] = torch.from_numpy(mask_file_res).float()
        
        #print("t4 - t3", t4 - t3)

        return output

    def train(self):
        self.mode = 'train'
        if self._aug_mode:
            self.aug_mode = True

    def eval(self):
        self.mode = 'val'
        self.aug_mode = False

    def test(self):
        self.mode = 'test'
        self.aug_mode = False
    
    def real(self):
        self.mode = "real"
        self.aug_mode = False

    @property
    def mode(self):
        return self._mode 

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'val', 'test', "real"]
        self._mode = value

    def load_data(self):
        dataset_dict = {"train" : self.train_dataset_dir,
                        "val" : self.val_dataset_dir,
                        "real" : self.real_dataset_dir,}
        splits = ["train", "val", "real"]
        data = defaultdict(list)
        for split in splits:
            if split == "train" or split == "val":
                iter = 0
                dataset_dir = dataset_dict[split]
                rgb_files = glob.glob(os.path.join(dataset_dir, "*", "*.npy"))
                rgb_files.sort()
                for rgb_file in tqdm(rgb_files, f"Loading {split} ..."):
                     # rgb_file like this : '/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/10443/0029_color.png'
                    scene_id, frame_name = rgb_file.split('/')[-2], rgb_file.split('/')[-1]
                    img_name = f"{scene_id}_{frame_name}"
                    
                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    depth_file = rgb_file
                    depth_pf_file = rgb_file
                    joints_file = rgb_file.replace('simDepthImage.npy', 'meta.json')
                    with open(joints_file, 'r') as fd:
                        json_data = json.load(fd)[0]
                    json_keypoints_data = json_data["keypoints"]                    
                    json_joints_data = json_data["joints_3n_fixed_42"]
                    
                    json_joints_8_data = json_data["joints"]
                    joints_8_pos = [kp["position"] for idx, kp in enumerate(json_joints_8_data)] 
                    json_keypoints_pos = [kp["location_wrt_cam"] for idx, kp in enumerate(json_keypoints_data)][:-2]
                    
                    R2C_Mat = json_keypoints_data[0]["R2C_mat"]
                    R2C_Trans = json_keypoints_data[0]["location_wrt_cam"]
                    joints_loc_wrt_cam_data = [json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))]
                    assert len(self.JOINT_NAMES) == len(joints_loc_wrt_cam_data)
    
                    joints_pos = np.zeros((len(self.JOINT_NAMES), 3), dtype=np.float32)
                    for idx, k in enumerate(self.JOINT_NAMES):
                        loc_wrt_cam = joints_loc_wrt_cam_data[idx]
                        joints_pos[idx] = [loc_wrt_cam[0], 
                                           loc_wrt_cam[1],
                                           loc_wrt_cam[2],] 
    
                    iter += 1
                    sample = {
                            'rgb_file': rgb_file,
                            "depth_file" : depth_file, # mm
                            "depth_pf_file" : depth_pf_file,
                            'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]
                            "joints_8" : joints_8_pos, 
                            "joints_kps" : json_keypoints_pos,
                            "R2C_Mat" :  R2C_Mat,
                            "R2C_Trans" : R2C_Trans
                        }
    
                    data[split].append(sample)
            
            elif split == "real":
                dataset_dir = dataset_dict[split]
                depth_files = glob.glob(os.path.join(dataset_dir, "*.npy"))  
                 
                print("length of dataset_dir", len(depth_files))
                depth_files.sort() 
                
                for depth_file in tqdm(depth_files, f"Loading {split} ..."):
                    # rgb_file like this : '/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/10443/0029_color.png'
                    frame_name = (depth_file.split('/')[-1]).replace(".npy", "")
                    
                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    meta_file = depth_file.replace('npy', 'json')
                    with open(meta_file, 'r') as fd:
                        json_data = json.load(fd)[0]
                    json_keypoints_data = json_data["keypoints"]
                    json_joints_data = json_data["joints_3n_fixed_42"]
                    json_joints_8_data = json_data["joints"]
                    
                    joints_8_pos = [kp["position"] for idx, kp in enumerate(json_joints_8_data)] 
                    json_keypoints_pos = [kp["location_wrt_cam"] for idx, kp in enumerate(json_keypoints_data)]
                    
                    R2C_Mat = json_keypoints_data[0]["R2C_mat"]
                    R2C_Trans = json_keypoints_data[0]["location_wrt_cam"]
                    joints_loc_wrt_cam_data = [json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))]
                    assert len(self.JOINT_NAMES) == len(joints_loc_wrt_cam_data)
    
                    joints_pos = np.zeros((len(self.JOINT_NAMES), 3), dtype=np.float32)
                    for idx, k in enumerate(self.JOINT_NAMES):
                        loc_wrt_cam = joints_loc_wrt_cam_data[idx]
                        joints_pos[idx] = [loc_wrt_cam[0], 
                                           loc_wrt_cam[1],
                                           loc_wrt_cam[2],] 
    
                    iter += 1
    
                    sample = {
                            "depth_file" : depth_file,        # mm
                            "rgb_file" : depth_file,
                            'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]
                            "joints_8" : joints_8_pos, 
                            "joints_kps" : json_keypoints_pos,
                            "R2C_Mat" :  R2C_Mat,
                            "R2C_Trans" : R2C_Trans
                        }
    
                    data[split].append(sample)
               
                    

        return data

if __name__ == "__main__":
    PandaDataset = Voxel_dataset(train_dataset_dir=Path("/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn/"),
                                            val_dataset_dir=Path("/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/"),
                                            real_dataset_dir=Path("/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test/depth/"),
                                            joint_names = JOINT_NAMES,
                                            run=[0],
                                            init_mode="train", 
                                            #noise=self.cfg['noise'],
                                            img_type="D",
                                            raw_img_size=tuple([640, 360]),
                                            input_img_size=tuple([384, 216]),
                                            sigma=3.0,
                                            norm_type="min_max",
                                            network_input="XYZ",
                                            network_task="3d_RPE",
                                            depth_range=[500, 3380, 15],
                                            depth_range_type="normal",
                                            aug_type="3d",
                                            aug_mode=True) 
                                       #demo=self.cfg['demo'])
    a = dict()
    a["train"] = copy.copy(PandaDataset)
    PandaDataset.train()
    a["val"] = PandaDataset
#    for key, value in a.items():
#        print(len(value))
    print("len(PandaDataset)", len(PandaDataset))
    for idx, item in enumerate(PandaDataset):
        pass
#        if idx >= 1: 
#            break
#        #'depth': torch.from_numpy(depth_img[None, ...]),
#        #    'depthvis': depth_img_vis,
#        #    'z_values': z_values,
#        #    'K_depth': intrinsic,
#        #    'joints_2D_depth': joints_2D,
#        #    'joints_3D_Z': joints_3D_Z,
#        cv2.imwrite("visual_depth.png", item["depthvis"])
#        rgb_path = item["rgb_path"]
#        img = cv2.imread(rgb_path)
#        img = cv2.resize(img, tuple([384, 216]), interpolation=cv2.INTER_NEAREST)
#        img = img[12:-12, :, :]
#        cv2.imwrite("visual_rgb.png", img)
#        
#        kp_projs = item["joints_2D_depth"]
#        images = []
#        for n in range(len(kp_projs)):
#            image = overlay_points_on_image("visual_rgb.png", [kp_projs[n]], annotation_color_dot = ["red"], point_diameter=4)
#            images.append(image)
#        
#        img = mosaic_images(
#                    images, rows=3, cols=4, inner_padding_px=10
#                )
#        save_path = f"./visual_rgb.png".replace("rgb", "joint_check")
#        img.save(save_path)
#        print("heatmap_25d", item['heatmap_25d'].shape)
#        
#        N, _, _ = item["heatmap_25d"].shape
#        heatmap_uv = item["heatmap_25d"][:N//2, :, :].detach().cpu().numpy()
#        heatmap_uz = item["heatmap_25d"][N//2:, :, :].detach().cpu().numpy()
#        
#        heatmap_uv = np.sum(heatmap_uv, axis=0)
#        heatmap_uz = np.sum(heatmap_uz, axis=0)
#        heatmap_uv = heatmap_uv[:, :, None] * 255
#        heatmap_uz = heatmap_uz[:, :, None] * 255
#        cv2.imwrite(f"./uv_hm.png", heatmap_uv)
#        cv2.imwrite(f"./uz_hm.png", heatmap_uz)
        
#        for j in range(item["heatmap_25d"].shape[0]):
#            this_img = item["heatmap_25d"][j]
#            #print(type(this_img))
#            cv2.imwrite(f"./{str(j).zfill(2)}_hm.png", this_img.detach().cpu().numpy()[:, :, None] * 255)
        
        
#        print("joints_2D_depth.shape", item["joints_2D_depth"])
#        print("joints_3D_Z.shape", item["joints_3D_Z"])
    #PandaDataset
    
    
    
    
    
    
    
    
    
    
    
    
