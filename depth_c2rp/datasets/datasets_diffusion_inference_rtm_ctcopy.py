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
from depth_c2rp.utils.mm_utils import TopdownAffine, GetBBoxCenterScale

JOINT_NAMES = [f"panda_joint_3n_{i+1}" for i in range(14)]

class Diff_dataset(Dataset):
    def __init__(self, train_dataset_dir: Path, val_dataset_dir: Path, real_dataset_dir: Path, joint_names: list, run: list, init_mode: str = 'train', img_type: str = 'D',
                 raw_img_size: tuple = (640, 360), input_img_size: tuple = (384, 384), output_img_size : tuple = (384, 384), sigma: int = 4., norm_type: str = 'mean_std', network_input: str = 'D', network_output: str = 'H', network_task: str = '3d_RPE', depth_range: tuple = (500, 3380, 15), depth_range_type: str = 'normal', aug_type: str = '3d', aug_mode: bool = True, noise: bool = False, demo: bool = False, load_mask: bool = False, mask_dict : dict = {}, hole_filling : bool = True, aug_intrin : bool = False, intrin_params : dict = {}, change_intrinsic : bool = False, cond_norm : bool = False, mean : list=[], std : list = [], aug_mask: bool = False, aug_mask_ed : bool = False, whole_flag: bool = False, hm_mode: str = "spdh_intno", kps_14_name: str="42", simcc_flag: bool=True, mid_epoch: int=35,
                 
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
        self.GetBBoxCenterScale = GetBBoxCenterScale()
        self.TopdownAffine = TopdownAffine(input_size=self.output_img_size)


    def __len__(self):
        return len(self.data[self.mode])

    def __getitem__(self, idx):
        t1 = time.time()
        sample = self.data[self.mode][idx].copy()

        if "RGB" in self.network_input:
            prev_rgb_img = cv2.imread(sample["prev_rgb_file"])
            next_rgb_img = cv2.imread(sample["next_rgb_file"])
            
        if "MASK" in self.network_input:
            prev_mask_img = cv2.imread(sample["prev_mask_file"], cv2.IMREAD_UNCHANGED)[:, :, 2:3]
            next_mask_img = cv2.imread(sample["next_mask_file"], cv2.IMREAD_UNCHANGED)[:, :, 2:3]
        
        if self.mode == "train" or self.mode == "val":
            prev_mask_img = (prev_mask_img < 1.0).astype(np.uint8)
            next_mask_img = (next_mask_img < 1.0).astype(np.uint8)
        elif self.mode == "real":
            prev_mask_img = (prev_mask_img > 0.0).astype(np.uint8)
            next_mask_img = (next_mask_img > 0.0).astype(np.uint8)
        
        if self.mode == "train" or self.mode == "val":
            cam_settings_data = json.load(open(str(self.train_dataset_dir / "_camera_settings.json"), 'r'))
        else:
            cam_settings_data = json.load(open(str(self.real_dataset_dir / "_camera_settings.json"), 'r'))

        scale = self.input_img_size[0] / self.raw_img_size[0]
        fx, fy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fx"], cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fy"] 
        cx, cy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cx"], cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cy"] 
        
        intrinsic = np.asarray([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])

        prev_joints_2D = np.zeros((sample['prev_joints'].shape[0], 2))
        next_joints_2D = np.zeros((sample['next_joints'].shape[0], 2))
        prev_joints_3D_Z = np.zeros((sample['prev_joints'].shape[0], 3))
        next_joints_3D_Z = np.zeros((sample['next_joints'].shape[0], 3))  
        for n, joint in enumerate(sample["prev_joints"]):
            point3d = np.asarray([joint[0], joint[1], joint[2]]) # right handed reference frame
            prev_joints_3D_Z[n] = point3d.copy()
            u, v, w = (intrinsic @ point3d[:, None]).T.squeeze(0)
            u = u / w
            v = v / w
            prev_joints_2D[n] = [u, v]
            
        for n, joint in enumerate(sample["next_joints"]):
            point3d = np.asarray([joint[0], joint[1], joint[2]]) # right handed reference frame
            next_joints_3D_Z[n] = point3d.copy()
            u, v, w = (intrinsic @ point3d[:, None]).T.squeeze(0)
            u = u / w
            v = v / w
            next_joints_2D[n] = [u, v]  
           
        next_joints_2D_visible = (next_joints_2D[:, 0] >= 0.0) * (next_joints_2D[:, 0] < self.output_img_size[0]) * (next_joints_2D[:, 1] >= 0) * (next_joints_2D[:, 1] < self.output_img_size[1])
        
        prev_col_info, prev_row_info, _ = np.nonzero(prev_mask_img)
        next_col_info, next_row_info, _ = np.nonzero(next_mask_img)
        
        min_col, max_col = np.min(np.array(0).astype(np.int64)), np.min(np.array(0).astype(np.int64))
        min_row, max_row = np.min(np.array(0).astype(np.int64)), np.min(np.array(0).astype(np.int64)) 
        
        if len(prev_col_info) > 0 and len(next_col_info) > 0:
            min_col = min(np.min(prev_col_info), np.min(next_col_info))
            max_col = max(np.max(prev_col_info), np.max(next_col_info))
                
        if len(prev_row_info) > 0 and len(next_row_info) > 0:
            min_row = min(np.min(prev_row_info), np.min(next_row_info))
            max_row = max(np.max(prev_row_info), np.max(next_row_info))
        
        bbox = [min_col, min_row, max_col + 1, max_row  + 1]    

        if "RGB" in self.network_input:
            results = dict()
            results["img"] = prev_rgb_img
            results["img_shape"] = prev_rgb_img.shape[:2]
            results["img_mask"] = prev_mask_img
            results["next_img"] = next_rgb_img
            results["next_img_mask"] = next_mask_img

            results["prev_keypoints"] = prev_joints_2D[None, ...]
            results["keypoints"] = next_joints_2D[None, ...]
            results["keypoints_visible"] = next_joints_2D_visible[None, ...]
            mm_bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            results["bbox"] = np.array(mm_bbox)[None, ...]

            results = self.GetBBoxCenterScale.transform(results)
            results = self.TopdownAffine.transform(results)

            prev_rgb_img_input = results["img"]
            prev_mask_img_input = results["img_mask"]
            prev_rgb_img_input = normalize_image(prev_rgb_img_input, mean=self.mean, std=self.std)
            prev_rgb_img_vis = ((prev_rgb_img_input * self.std) + self.mean ) * 255
            
            next_rgb_img_input = results["next_img"]
            next_mask_img_input = results["next_img_mask"]
            next_rgb_img_input = normalize_image(next_rgb_img_input, mean=self.mean, std=self.std)
            next_rgb_img_vis = ((next_rgb_img_input * self.std) + self.mean ) * 255            
            
            trans_output = results["trans_output"]
            trans_output_inverse = results["trans_output_inverse"]
            
            prev_joints_2D_output = results["prev_transformed_keypoints"][0]
            next_joints_2D_output = results["transformed_keypoints"][0]
            next_joints_2D_output_floor = np.floor(next_joints_2D_output)
            next_joints_2D_output_off = next_joints_2D_output - next_joints_2D_output_floor
            
            # keypoint_x_labels = results["keypoint_x_labels"].squeeze(0)
            # keypoint_y_labels = results["keypoint_y_labels"].squeeze(0)
            # keypoint_weights = results["keypoint_weights"].squeeze(0)


        if "BBOX" in self.network_input:
            pass            
        elif "CONCAT" in self.network_input:
            prev_rgb_img_input = np.concatenate([prev_rgb_img_input, prev_mask_img_input[..., None]], axis=-1)
            next_rgb_img_input = np.concatenate([next_rgb_img_input, next_mask_img_input[..., None]], axis=-1)  
        else:
            #print("!!!")
            prev_rgb_img_input *= (prev_mask_img_input[..., None] > 0.0)
            next_rgb_img_input *= (next_mask_img_input[..., None] > 0.0)
            
            prev_rgb_img_vis *= (prev_mask_img_input[..., None] > 0.0)
            next_rgb_img_vis *= (next_mask_img_input[..., None] > 0.0)   

        R2C_Mat_after_aug = np.array(sample["next_R2C_Mat"]) # 3 x 3
        R2C_Trans_after_aug = np.array(sample["next_R2C_Trans"]) # 3
        R2C_Pose_after_aug = np.eye(4, 4)
        R2C_Pose_after_aug[:3, :3] = R2C_Mat_after_aug
        R2C_Pose_after_aug[:3, 3] = R2C_Trans_after_aug 
        next_joints_3D_Z_rob = (next_joints_3D_Z - R2C_Pose_after_aug[:3, 3:].T) @ R2C_Pose_after_aug[:3, :3]

        if self.cond_norm:
            #noise = 3.0 * np.random.randn(joints_2D.shape[0], joints_2D.shape[1])
            next_input_joints_2D = np.zeros_like(next_joints_2D) 
            #print("noise", noise)
            next_input_joints_2D[:, 0] = (next_joints_2D[:, 0]  - intrinsic[0, 2]) / intrinsic[0, 0]
            next_input_joints_2D[:, 1] = (next_joints_2D[:, 1]  - intrinsic[1, 2]) / intrinsic[1, 1]
        else:
            raise NotImplementedError         

        output = {
            "rgb_path" : sample['next_rgb_file'],
            "trans_output": trans_output.astype(np.float64),
            "trans_output_inverse": trans_output_inverse.astype(np.float64),
            "prev_joints_3D_Z": prev_joints_3D_Z.astype(np.float64),
            "next_joints_3D_Z": next_joints_3D_Z.astype(np.float64),
            "next_joints_3D_Z_rob": next_joints_3D_Z_rob,
            "prev_joints_2D": prev_joints_2D.astype(np.float64),
            "prev_joints_2D_output": prev_joints_2D_output.astype(np.float64),
            "next_joints_2D": next_joints_2D.astype(np.float64),
            "next_joints_2D_output": next_joints_2D_output.astype(np.float64),
            "bbox" : np.array(bbox).astype(np.float64),
            "intrinsic": intrinsic,
            "next_joints_2D_cond_norm": next_input_joints_2D,
            "next_joints_7" : torch.from_numpy(np.array(sample["next_joints_8"])[:7]).float(),
            "next_joints_3D_kps" : torch.from_numpy(np.array(sample["next_joints_kps"]))[[0,2,3,4,6,7,8]],
            "next_R2C_Pose" : R2C_Pose_after_aug[:3, :],
        }

        if "RGB" in self.network_input:
            output["prev_rgb_img_input"] = (prev_rgb_img_input.transpose(2, 0, 1)).astype(np.float64)
            output["prev_vis"] = prev_rgb_img_vis.astype(np.float64)
            output["next_rgb_img_input"] = (next_rgb_img_input.transpose(2, 0, 1)).astype(np.float64)
            output["next_vis"] = next_rgb_img_vis.astype(np.float64)  

        # if self.simcc_flag:
        #     output["keypoint_x_labels"] = keypoint_x_labels
        #     output["keypoint_y_labels"] = keypoint_y_labels
        #     output["keypoint_weights"] = keypoint_weights

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
    
    # def pred_mask(self, pred_mask_flag):
    #     if pred_mask_flag != "":
    #         self.pred_mask_flag = pred_mask_flag
    
    @property
    def mode(self):
        return self._mode 

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'val', 'real']
        self._mode = value


    def load_data(self):
        dataset_dict = {"train" : self.train_dataset_dir,
                        "val" : self.val_dataset_dir,
                        "real" : self.real_dataset_dir,
                        }
        print(self.real_dataset_dir)
        splits = ["train", "val", "real"]
        data = defaultdict(list)

        

        for split in splits:
            if split == "train" or split == "val":
                iter = 0
                dataset_dir = dataset_dict[split]
                rgb_files = glob.glob(os.path.join(dataset_dir, "*", "*_color.png"))
                rgb_files.sort()

                for rgb_file in tqdm(rgb_files, f"Loading {split} ..."):
                    # rgb_file like this : '/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/10443/0029_color.png'
                    this_split = rgb_file.split('/')
                    scene_id, frame_name = this_split[-2], this_split[-1][:4]
                    img_name = f"{scene_id}_{frame_name}"
                    frame_id = int(frame_name)
                    
                    if frame_id % 3 == 0 or frame_id % 3 == 1:
                        next_frame_name = str(frame_id + 1).zfill(4)
                    elif frame_id % 3 == 2:
                        next_frame_name = str(frame_id).zfill(4)
                        new_frame_id = str(frame_id - 2).zfill(4)
                        rgb_file = ('/').join(this_split[:-1] + [new_frame_id + "_color.png"])
                    next_rgb_file = ('/').join(this_split[:-1] + [next_frame_name + "_color.png"])
                    if not os.path.exists(next_rgb_file) or not os.path.exists(rgb_file):
                        print(next_rgb_file)
                        continue
                    

                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    depth_file = rgb_file.replace('color.png', 'simDepthImage.npy')
                    joints_file = rgb_file.replace('color.png', 'meta.json')
                    mask_file = rgb_file.replace("color.png", "mask.exr")
                    with open(joints_file, 'r') as fd:
                        json_data = json.load(fd)[0]
                    json_keypoints_data = json_data["keypoints"]
                    json_joints_data = json_data[f"joints_3n_fixed_{self.kps_14_name}"]
                    keypoints_r2c_data = [json_keypoints_data[idx]["R2C_mat"] for idx in range(len(json_keypoints_data))]
                    joints_pos = np.array([json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))])
                    assert len(JOINT_NAMES) == len(joints_pos)
                    
                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    next_depth_file = next_rgb_file.replace('color.png', 'simDepthImage.npy')
                    next_joints_file = next_rgb_file.replace('color.png', 'meta.json')
                    next_mask_file = next_rgb_file.replace("color.png", "mask.exr")
                    with open(next_joints_file, 'r') as next_fd:
                        next_json_data = json.load(next_fd)[0]
                    next_json_keypoints_data = next_json_data["keypoints"]
                    next_json_joints_data = next_json_data[f"joints_3n_fixed_{self.kps_14_name}"]
                    next_keypoints_r2c_data = [next_json_keypoints_data[idx]["R2C_mat"] for idx in range(len(next_json_keypoints_data))]
                    next_joints_pos = np.array([next_json_joints_data[idx]["location_wrt_cam"] for idx in range(len(next_json_joints_data))])
                    assert len(JOINT_NAMES) == len(next_joints_pos)     
                    next_R2C_Mat = next_json_keypoints_data[0]["R2C_mat"]
                    next_R2C_Trans = next_json_keypoints_data[0]["location_wrt_cam"]    
                    next_json_joints_8_data = next_json_data["joints"]
                    next_joints_8_pos = [kp["position"] for idx, kp in enumerate(next_json_joints_8_data)]            
                    next_json_keypoints_pos = [kp["location_wrt_cam"] for idx, kp in enumerate(next_json_keypoints_data)]                              
                    iter += 1
    
                    sample = {
                            "prev_rgb_file": rgb_file,
                            "prev_depth_file" : depth_file, # mm
                            "prev_mask_file"  : mask_file,
                            "prev_joints": joints_pos, 
                            "next_rgb_file": next_rgb_file,
                            "next_depth_file" : next_depth_file, # mm
                            "next_mask_file"  : next_mask_file,
                            "next_joints": next_joints_pos,      
                            "next_R2C_Mat" :  next_R2C_Mat,
                            "next_R2C_Trans" : next_R2C_Trans,     
                            "next_joints_8" : next_joints_8_pos,     
                            "next_joints_kps" : next_json_keypoints_pos,
                        }
                    #print("sample", sample)
    
                    data[split].append(sample)
            elif split == "real":
                iter = 0
                dataset_dir = dataset_dict[split]
                print("dataset_dir", dataset_dir)
                rgb_files = glob.glob(os.path.join(dataset_dir, "*.png"))  
                print("length of dataset_dir", len(rgb_files))
                rgb_files.sort()
                #print("rgb_files", rgb_files)

                if "PRED_MASK" in self.network_input:
                    self.pred_mask_flag = "_ours_009_mask.exr"
                
                for rgb_file in tqdm(rgb_files[:-1], f"Loading {split} ..."):
                    this_split = rgb_file.split('/')
                    frame_name = this_split[-1][:6]
                    frame_id = int(frame_name)
                    
                    next_frame_name = str(frame_id + 1).zfill(6)
                    next_rgb_file = ('/').join(this_split[:-1] + [next_frame_name + ".png"])
                    if not os.path.exists(next_rgb_file):
                        print(next_rgb_file)
                        continue
                    
                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    depth_file = rgb_file.replace('png', 'npy')
                    joints_file = rgb_file.replace('png', 'json')
                    mask_file = rgb_file.replace(".png", self.pred_mask_flag)
                    with open(joints_file, 'r') as fd:
                        json_data = json.load(fd)[0]
                    json_keypoints_data = json_data["keypoints"]
                    json_joints_data = json_data[f"joints_3n_fixed_{self.kps_14_name}"]
                    keypoints_r2c_data = [json_keypoints_data[idx]["R2C_mat"] for idx in range(len(json_keypoints_data))]
                    joints_pos = np.array([json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))])
                    assert len(JOINT_NAMES) == len(joints_pos)
                    
                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    next_depth_file = next_rgb_file.replace('png', 'npy')
                    next_joints_file = next_rgb_file.replace('png', 'json')
                    next_mask_file = next_rgb_file.replace(".png", self.pred_mask_flag)
                    with open(next_joints_file, 'r') as next_fd:
                        next_json_data = json.load(next_fd)[0]
                    next_json_keypoints_data = next_json_data["keypoints"]
                    next_json_joints_data = next_json_data[f"joints_3n_fixed_{self.kps_14_name}"]
                    next_keypoints_r2c_data = [next_json_keypoints_data[idx]["R2C_mat"] for idx in range(len(next_json_keypoints_data))]
                    next_joints_pos = np.array([next_json_joints_data[idx]["location_wrt_cam"] for idx in range(len(next_json_joints_data))])
                    assert len(JOINT_NAMES) == len(next_joints_pos) 
                    next_R2C_Mat = next_json_keypoints_data[0]["R2C_mat"]
                    next_R2C_Trans = next_json_keypoints_data[0]["location_wrt_cam"]   
                    next_json_joints_8_data = next_json_data["joints"]
                    next_joints_8_pos = [kp["position"] for idx, kp in enumerate(next_json_joints_8_data)]    
                    next_json_keypoints_pos = [kp["location_wrt_cam"] for idx, kp in enumerate(next_json_keypoints_data)]                              
                    

                    sample = {
                            "prev_rgb_file": rgb_file,
                            "prev_depth_file" : depth_file, # mm
                            "prev_mask_file"  : mask_file,
                            "prev_joints": joints_pos, 
                            "next_rgb_file": next_rgb_file,
                            "next_depth_file" : next_depth_file, # mm
                            "next_mask_file"  : next_mask_file,
                            "next_joints": next_joints_pos,     
                            "next_R2C_Mat" :  next_R2C_Mat,
                            "next_R2C_Trans" : next_R2C_Trans,            
                            "next_joints_8" : next_joints_8_pos, 
                            "next_joints_kps" : next_json_keypoints_pos,
                        }                          
                    
   
                    data[split].append(sample)
                    
        self.data = data
        return data      



