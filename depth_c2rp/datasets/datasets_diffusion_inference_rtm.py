import json
from collections import defaultdict
from pathlib import Path

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import os
import copy
import sys
import pickle5 as pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyquaternion import Quaternion
import glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from depth_c2rp.utils.rgb_utils import get_affine_transform, _get_input, affine_transform_and_clip, affine_transform_pts, normalize_image
from mmpose.codecs.simcc_label import SimCCLabel

JOINT_NAMES = [f"panda_joint_3n_{i+1}" for i in range(14)]

def init_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class Diff_dataset(Dataset):
    def __init__(self, train_dataset_dir: Path, val_dataset_dir: Path, real_dataset_dir: Path, joint_names: list, run: list, init_mode: str = 'train', img_type: str = 'D',
                 raw_img_size: tuple = (640, 360), input_img_size: tuple = (384, 384), output_img_size : tuple = (384, 384), sigma: int = 4., norm_type: str = 'mean_std', network_input: str = 'D', network_output: str = 'H', network_task: str = '3d_RPE', depth_range: tuple = (500, 3380, 15), depth_range_type: str = 'normal', aug_type: str = '3d', aug_mode: bool = True, noise: bool = False, demo: bool = False, load_mask: bool = False, mask_dict : dict = {}, hole_filling : bool = True, aug_intrin : bool = False, intrin_params : dict = {}, change_intrinsic : bool = False, cond_norm : bool = False, mean : list=[], std : list = [], aug_mask: bool = False, aug_mask_ed : bool = False, whole_flag: bool = False, hm_mode: str = "spdh_intno", kps_14_name: str="42", simcc_flag: bool=True, mid_epoch: int=35,
                 rel: bool=False,
                 ):
        """Load Baxter robot synthetic dataset

        Parameters
        ----------
        dataset_dir: Path
            Dataset path.
        run: list
            Synthetic run to load (0, 1, or both).
        init_mode: str
            Loading mode (train -> train/val set, test -> test set).
        img_type: str
            Type of image to load (RGB, D or RGBD).
        img_size: str
            Image dimensions to which resize dataset images.
        sigma: int
            Variance of ground truth gaussian heatmaps.
        norm_type: str
            Type of normalization (min_max, mean_std supported).
        network_input: str
            Input type of the network (D, RGB, RGBD, XYZ).
        network_task: str
            Task the network should solve (2d or 3d robot pose estimation).
        ref_frame: str
            Reference frame of gt joints (right or left handed).
        surrounding_removals: bool
            Activate or deactivate removal of surrounding walls/objects.
        aug_type: str
            Type of data augmentation on the input (2d or 3d).
        aug_mode: bool
            Activate or deactivate augmentation during training.
        noise: bool
            If true, adds random pepper noise to data augmentation during training.
        demo: bool
            Useful for loading a portion of the dataset when debugging.
        """
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
        self.output_img_size = output_img_size
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
        self.hole_filling = hole_filling
        self.aug_intrin = aug_intrin
        self.intrin_params = intrin_params
        self.change_intrinsic = change_intrinsic
        self.cond_norm = cond_norm
        self.mean = mean
        self.std = std
        self.aug_mask = aug_mask
        self.aug_mask_ed = aug_mask_ed
        self.whole_flag = whole_flag
        self.hm_mode = hm_mode
        self.kps_14_name = kps_14_name
        self.curr_epoch = 0
        self.mid_epoch = mid_epoch
        self.simcc_flag = simcc_flag
        codec =dict(
                    input_size=self.output_img_size,
                    sigma=(6., 6.93),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)
        self.simcc_label = SimCCLabel(**codec)
        self.data = self.load_data()
        self.rel = rel
        

    def __len__(self):
        return len(self.data[self.mode])
    
    def __getitem__(self, idx):
        t1 = time.time()
        sample = self.data[self.mode][idx].copy()
        # image loading (depth and/or RGB)
        # depth_img = cv2.imread(sample['depth_file'], cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float64) * 1000 # milimeters 
        if self.mode == "train":
            if "XYZ" in self.network_input:
                depth_img = np.load(sample["depth_file"]).astype(np.float32) * 1000
            elif "RGB" in self.network_input:
                rgb_img = cv2.imread(sample["rgb_file"])
        elif self.mode == "real"  or self.mode == "val":
            if "XYZ" in self.network_input:
                depth_img = np.load(sample["depth_file"]).astype(np.float32)
            elif "RGB" in self.network_input:
                rgb_img = cv2.imread(sample["rgb_file"])
        
        if "MASK" in self.network_input:
            mask_img = cv2.imread(sample["mask_file"], cv2.IMREAD_UNCHANGED)
            if "PRED_MASK" in self.network_input and self.mode == "real":
                mask_img = cv2.imread(sample["pred_mask_file"], cv2.IMREAD_UNCHANGED)
            mask_img = mask_img[:, :, 2:3]
            #print(np.min(mask_img))
            if self.mode == "real":
                mask_img = (mask_img > 0.0).astype(np.float64)
            else:  
                mask_img = (mask_img < 1.0).astype(np.float64)
        else:
            mask_img = None
        
        # initialize intrinsic matrix
        if self.mode == "train":
            cam_settings_data = json.load(open(str(self.train_dataset_dir / "_camera_settings.json"), 'r'))
        elif self.mode == "val":
            cam_settings_data = json.load(open(str(self.val_dataset_dir / "_camera_settings.json"), 'r'))
        elif self.mode == "real":
            cam_settings_data = json.load(open(str(self.real_dataset_dir / "_camera_settings.json"), 'r'))
            
        scale = self.input_img_size[0] / self.raw_img_size[0]
        fx, fy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fx"], cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fy"] 
        cx, cy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cx"], cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cy"]
        
        intrinsic = np.asarray([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        
        joints_2D = np.zeros((sample['joints'].shape[0], 2))
        joints_3D_Z = np.zeros((sample['joints'].shape[0], 3)) 
        for n, joint in enumerate(sample["joints"]):
            point3d = np.asarray([joint[0], joint[1], joint[2]]) # right handed reference frame
            joints_3D_Z[n] = point3d.copy()
            u, v, w = (intrinsic @ point3d[:, None]).T.squeeze(0)
            u = u / w
            v = v / w
            joints_2D[n] = [u, v] 
        
        joints_2D_visible = (joints_2D[:, 0] >= 0.0) * (joints_2D[:, 0] < self.output_img_size[0]) * (joints_2D[:, 1] >= 0) * (joints_2D[:, 1] < self.output_img_size[1])
        
        col_info, row_info, _ = np.nonzero(mask_img)
        min_col, max_col = np.min(np.array(0).astype(np.int64)), np.min(np.array(0).astype(np.int64))
        min_row, max_row = np.min(np.array(0).astype(np.int64)), np.min(np.array(0).astype(np.int64)) 
        
        if len(col_info) > 0:
            min_col = np.min(col_info)
            max_col = np.max(col_info)
                
        if len(row_info) > 0:
            min_row = np.min(row_info)
            max_row = np.max(row_info)
        
        bbox = [min_col, min_row, max_col + 1, max_row  + 1]
        
        rgb_img = rgb_img[bbox[0] : bbox[2], bbox[1] : bbox[3], :] # H x W x 3
        mask_img = mask_img[bbox[0] : bbox[2], bbox[1] : bbox[3], :] # H x W x 1
        
        #rgb_img *= (mask_img > 0.0)
        
        joints_2D_crop = np.zeros_like(joints_2D)
        joints_2D_crop[:, 0] = joints_2D[:, 0] - bbox[1]
        joints_2D_crop[:, 1] = joints_2D[:, 1] - bbox[0]
        
        crop_h, crop_w = rgb_img.shape[0], rgb_img.shape[1]
        #print("shape", rgb_img.shape)
        #print("intrinsic", intrinsic)
        c = np.array([crop_w / 2., crop_h / 2.], dtype=np.float64) # (crop_w/2, crop_h/2)
        s = max(crop_h, crop_w) * 1.0
        rot = 0.0
        
        trans_input = get_affine_transform(c, s, rot, [self.input_img_size[0], self.input_img_size[1]])
        trans_output = get_affine_transform(c, s, rot, [self.output_img_size[0], self.output_img_size[1]])
        trans_input_inverse = get_affine_transform(c, s, rot, [self.input_img_size[0], self.input_img_size[1]], inv=1)
        trans_output_inverse = get_affine_transform(c, s, rot, [self.output_img_size[0], self.output_img_size[1]], inv=1)
        
        rgb_img_input = _get_input(rgb_img, trans_input, self.input_img_size[0], self.input_img_size[1], interpolation_method=cv2.INTER_LINEAR)
        mask_img_input = _get_input(mask_img, trans_input, self.input_img_size[0], self.input_img_size[1], interpolation_method=cv2.INTER_NEAREST)
        joints_2D_input = affine_transform_pts(joints_2D_crop, trans_input)
        joints_2D_output = affine_transform_pts(joints_2D_crop, trans_output)
        joints_2D_output_floor = np.floor(joints_2D_output)
        joints_2D_output_off = joints_2D_output - joints_2D_output_floor
            
        #rgb_img_input = color_dithering(rgb_img_input.astype(np.float64), mode=self.mode)
        rgb_img_input = normalize_image(rgb_img_input, mean=self.mean, std=self.std)
        rgb_img_vis = ((rgb_img_input * self.std) + self.mean ) * 255
        
        joints_2D_output_visible = (joints_2D_output[:, 0] >= 0.0) * (joints_2D_output[:, 0] < self.output_img_size[0]) * (joints_2D_output[:, 1] >= 0) * (joints_2D_output[:, 1] < self.output_img_size[1])
        joints_2D_output_simcc = self.simcc_label.encode(keypoints=joints_2D_output[None, ...], keypoints_visible=joints_2D_output_visible[None, ...])
                
        keypoint_x_labels = joints_2D_output_simcc["keypoint_x_labels"].squeeze(0)
        keypoint_y_labels = joints_2D_output_simcc["keypoint_y_labels"].squeeze(0)
        keypoint_weights = joints_2D_output_simcc["keypoint_weights"].squeeze(0)
        
        joints_2D_output_ind = joints_2D_output_floor[:, 1:2] * self.output_img_size[1] + joints_2D_output_floor[:, 0:1]
        joints_2D_output_ind[joints_2D_output_ind >= (self.output_img_size[1] *self.output_img_size[0])] *= 0.0
        joints_2D_output_ind[joints_2D_output_ind < 0.0] *= 0.0
        
        if "BBOX" in self.network_input:
            rgb_img_input = rgb_img_input
        elif "CONCAT" in self.network_input:
            rgb_img_input = np.concatenate([rgb_img_input, mask_img_input[..., None]], axis=-1)
        else:
            rgb_img_input *= (mask_img_input[..., None] > 0.0)
        
        R2C_Mat_after_aug = np.array(sample["R2C_Mat"]) # 3 x 3
        R2C_Trans_after_aug = np.array(sample["R2C_Trans"]) # 3
        R2C_Pose_after_aug = np.eye(4, 4)
        R2C_Pose_after_aug[:3, :3] = R2C_Mat_after_aug
        R2C_Pose_after_aug[:3, 3] = R2C_Trans_after_aug 
        
        joints_3D_Z_rob = (joints_3D_Z - R2C_Pose_after_aug[:3, 3:].T) @ R2C_Pose_after_aug[:3, :3]

        
        
        if self.cond_norm:
            #noise = 3.0 * np.random.randn(joints_2D.shape[0], joints_2D.shape[1])
            input_joints_2D = np.zeros_like(joints_2D) 
            #print("noise", noise)
            input_joints_2D[:, 0] = (joints_2D[:, 0]  - intrinsic[0, 2]) / intrinsic[0, 0]
            input_joints_2D[:, 1] = (joints_2D[:, 1]  - intrinsic[1, 2]) / intrinsic[1, 1]
        else:
            raise NotImplementedError
        
        output = {
            'joints_3D_Z': joints_3D_Z.astype(np.float64),
            "joints_3D_kps" : torch.from_numpy(np.array(sample["joints_kps"]))[[0,2,3,4,6,7,8]],
            "joints_3D_Z_rob": joints_3D_Z_rob,
            "joints_7" : torch.from_numpy(np.array(sample["joints_8"])[:7]).float(),
            "R2C_Pose" : R2C_Pose_after_aug[:3, :],
            "rgb_path" : sample['rgb_file'],
            "meta_path" : sample["meta_file"],
            "trans_output": trans_output.astype(np.float64),
            "trans_output_inverse": trans_output_inverse.astype(np.float64),
            'joints_2D_cond_norm': input_joints_2D,
            'joints_2D': joints_2D.astype(np.float64),
            "joints_2D_output": joints_2D_output.astype(np.float64),
            "joints_2D_output_floor" : joints_2D_output_floor.astype(np.float64),
            "joints_2D_output_off" : joints_2D_output_off.astype(np.float64),
            "joints_2D_output_ind" : joints_2D_output_ind,
            "bbox" : np.array(bbox).astype(np.float64),
            "intrinsic": intrinsic,
        }
        
        if "RGB" in self.network_input:
            output["input_tensor"] = (rgb_img_input.transpose(2, 0, 1)).astype(np.float64)
            output["vis"] = rgb_img_vis.astype(np.float64)
        
        output["keypoint_x_labels"] = keypoint_x_labels
        output["keypoint_y_labels"] = keypoint_y_labels
        output["keypoint_weights"] = keypoint_weights

        if self.rel:
            joints_3D_Z_rel = joints_3D_Z.copy()
            joints_3D_Z_rel[1:, :] -= joints_3D_Z_rel[0:1, :]
            output["joints_3D_Z_rel"] = joints_3D_Z_rel
        
        return output

    def train(self):
        self.mode = 'train'
        if self._aug_mode:
            self.aug_mode = True
        self.curr_epoch += 1
        

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
        assert value in ['train', 'val', 'real']
        self._mode = value
    
    def load_data(self):
        self.dataset_dict = {"train" : self.train_dataset_dir,
                        "val" : self.val_dataset_dir,
                        "real" : self.real_dataset_dir,}
        splits = ["train", "val", "real"]
        data = defaultdict(list)
        for split in splits:
            if split == "train" or split == "val":
                iter = 0
                dataset_dir = self.dataset_dict[split]
                meta_files = glob.glob(os.path.join(dataset_dir, "*.pkl"))
                if len(meta_files) == 0:
                    meta_files = glob.glob(os.path.join(dataset_dir, "*", "*meta.json"))
                meta_files.sort()
                
                for meta_file in tqdm(meta_files[:1000], f"Loading {split} ..."):
                    rgb_file, depth_pf_file, depth_file = 0.0, 0.0, 0.0
                    
                    
                    # rgb_file, depth_pf_file, depth_file
                    try:
                        img_name = meta_file.replace(".pkl", "")
                        with open(meta_file , "rb") as fh:
                            json_data = pickle.load(fh)[0]
                    except: 
                        with open(meta_file, 'r') as fd:
                            json_data = json.load(fd)[0]
                    
                    # link kps info
                    json_keypoints_data = json_data["keypoints"]
                    json_keypoints_pos = [kp["location_wrt_cam"] for idx, kp in enumerate(json_keypoints_data)]
                    R2C_Mat = json_keypoints_data[0]["R2C_mat"]
                    R2C_Trans = json_keypoints_data[0]["location_wrt_cam"]
                    
                    # 14 kps info                
                    json_joints_data = json_data["joints_3n_fixed_42"]
                    joints_loc_wrt_cam_data = [json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))]
                    assert len(self.JOINT_NAMES) == len(joints_loc_wrt_cam_data)
                    
                    # joints info
                    json_joints_8_data = json_data["joints"]
                    joints_8_pos = [kp["position"] for idx, kp in enumerate(json_joints_8_data)] 
                    
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
                            "meta_file" : meta_file,
                            'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]
                            "joints_8" : joints_8_pos, 
                            "joints_kps" : json_keypoints_pos,
                            "R2C_Mat" :  R2C_Mat,
                            "R2C_Trans" : R2C_Trans
                        }
                    data[split].append(sample)
            
            elif split == "real":
                dataset_dir = self.dataset_dict[split]
                rgb_files = glob.glob(os.path.join(dataset_dir, "*.png"))  
                 
                print("length of dataset_dir", len(rgb_files))
                
                if len(rgb_files) == 0:
                    rgb_files = glob.glob(os.path.join(dataset_dir, "*.rgb.jpg"))  
                rgb_files.sort() 
                
                  
                for rgb_file in tqdm(rgb_files, f"Loading {split} ..."):
                    # rgb_file like this : '/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/10443/0029_color.png'
                    frame_name = (rgb_file.split('/')[-1]).replace(".png", "")
                    
                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    try:
                        depth_file = rgb_file.replace('png', 'npy')
                        meta_file = depth_file.replace('npy', 'json') 
                        mask_file = depth_file.replace(".npy", "_mask.exr")
                        pred_mask_file = depth_file.replace(".npy", "_ours_009_mask.exr")
                        with open(meta_file, 'r') as fd:
                            json_data = json.load(fd)[0]
                    except:
                        depth_file = rgb_file
                        meta_file = rgb_file.replace(".rgb.jpg", ".pkl") 
                        mask_file = rgb_file.replace(".rgb.jpg", "_mask.exr")
                        pred_mask_file = rgb_file.replace(".rgb.jpg", "_ours_009_mask.exr")
                        with open(meta_file, 'rb') as fd:
                            json_data = pickle.load(fd)[0]
                    
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
                            "rgb_file" : rgb_file,
                            "mask_file" : mask_file,
                            "pred_mask_file" : pred_mask_file, 
                            "meta_file" : meta_file,
                            'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]
                            "joints_8" : joints_8_pos, 
                            "joints_kps" : json_keypoints_pos,
                            "R2C_Mat" :  R2C_Mat,
                            "R2C_Trans" : R2C_Trans
                        }
    
                    data[split].append(sample)
            
            
#                dataset_dir = self.dataset_dict[split]
#                depth_files = glob.glob(os.path.join(dataset_dir, "*.npy"))  
#                 
#                print("length of dataset_dir", len(depth_files))
#                depth_files.sort() 
#                
#                for depth_file in tqdm(depth_files, f"Loading {split} ..."):
#                    # rgb_file like this : '/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/10443/0029_color.png'
#                    frame_name = (depth_file.split('/')[-1]).replace(".npy", "")
#                    
#                    # Here the dataset requires depth 8 and depth 16, so how to do that?
#                    rgb_file = depth_file.replace('npy', 'png')
#                    meta_file = depth_file.replace('npy', 'json') 
#                    mask_file = depth_file.replace(".npy", "_mask.exr")
#                    pred_mask_file = depth_file.replace(".npy", "_ours_009_mask.exr")
#                    with open(meta_file, 'r') as fd:
#                        json_data = json.load(fd)[0]
#                    json_keypoints_data = json_data["keypoints"]
#                    json_joints_data = json_data["joints_3n_fixed_42"]
#                    json_joints_8_data = json_data["joints"]
#                    
#                    joints_8_pos = [kp["position"] for idx, kp in enumerate(json_joints_8_data)] 
#                    json_keypoints_pos = [kp["location_wrt_cam"] for idx, kp in enumerate(json_keypoints_data)]
#                    
#                    R2C_Mat = json_keypoints_data[0]["R2C_mat"]
#                    R2C_Trans = json_keypoints_data[0]["location_wrt_cam"]
#                    joints_loc_wrt_cam_data = [json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))]
#                    assert len(self.JOINT_NAMES) == len(joints_loc_wrt_cam_data)
#    
#                    joints_pos = np.zeros((len(self.JOINT_NAMES), 3), dtype=np.float32)
#                    for idx, k in enumerate(self.JOINT_NAMES):
#                        loc_wrt_cam = joints_loc_wrt_cam_data[idx]
#                        joints_pos[idx] = [loc_wrt_cam[0], 
#                                           loc_wrt_cam[1],
#                                           loc_wrt_cam[2],] 
#    
#                    iter += 1
#    
#                    sample = {
#                            "depth_file" : depth_file,        # mm
#                            "rgb_file" : rgb_file,
#                            "mask_file" : mask_file,
#                            "pred_mask_file" : pred_mask_file, 
#                            "meta_file" : meta_file,
#                            'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]
#                            "joints_8" : joints_8_pos, 
#                            "joints_kps" : json_keypoints_pos,
#                            "R2C_Mat" :  R2C_Mat,
#                            "R2C_Trans" : R2C_Trans
#                        }
#    
#                    data[split].append(sample)
               
                    

        return data








