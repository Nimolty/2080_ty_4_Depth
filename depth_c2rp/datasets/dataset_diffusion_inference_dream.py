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
import pytorch3d
import pickle5 as pickle
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" 

from depth_c2rp.utils.spdh_utils import augment_3d, depthmap2pointcloud, depthmap2points, overlay_points_on_image, mosaic_images, pointcloud2depthmap, hole_filling, nearest_hole_filling, augment_3d_diff
from depth_c2rp.utils.spdh_utils import apply_depth_normalization_16bit_image, heatmap_from_kpoints_array, gkern, compute_rigid_transform, augment_affine

JOINT_NAMES = [f"panda_joint_3n_{i+1}" for i in range(14)]


def init_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


class Diff_dataset(Dataset):
    def __init__(self, train_dataset_dir: Path, val_dataset_dir : Path, real_dataset_dir : Path, joint_names: list, run: list, init_mode: str = 'train', img_type: str = 'D',
                 raw_img_size: tuple = (640, 360), input_img_size: tuple = (384, 216), sigma: int = 4., norm_type: str = 'mean_std',
                 network_input: str = 'D', network_output: str = 'H', network_task: str = '3d_RPE',
                 depth_range: tuple = (500, 3380, 15), depth_range_type: str = 'normal', aug_type: str = '3d',
                 aug_mode: bool = True, noise: bool = False, demo: bool = False, mask_dict : dict = {}, unnorm_depth: bool=False, cx_delta: int = 0, cy_delta: int = 0, change_intrinsic: bool=False, uv_input : bool = False, intrin_aug_params : dict = {}, cond_uv_std : float=0.0, large_cond_uv_std : float=0.0, prob_large_cond_uv : float=0.0, cond_norm : bool = False, mean : list=[], std : list = [],):
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
        self.mask_dict = mask_dict
        self.unnorm_depth = unnorm_depth
        self.change_intrinsic = change_intrinsic
        self.uv_input = uv_input
        self.intrin_aug_params = intrin_aug_params
        self.cond_uv_std = cond_uv_std
        self.prob_large_cond_uv = prob_large_cond_uv
        self.large_cond_uv_std = large_cond_uv_std
        self.cond_norm = cond_norm
        self.mean = mean
        self.std = std
        self.data = self.load_data()

    def __len__(self):
        return len(self.data[self.mode])

    def __getitem__(self, idx):    
        t1 = time.time()
        sample = self.data[self.mode][idx].copy()
        
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
            mask_img = mask_img[:, :, 2]
            #print(np.min(mask_img))
            mask_img = (mask_img > 0.0).astype(np.float64)
        else:
            mask_img = None
        
        if "XYZ" in self.network_input:
            depth_img = depth_img * mask_img
        elif "RGB" in self.network_input:
            rgb_img = rgb_img * mask_img[..., None]  
        
            
        # RGB and depth scale
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
        
        # adapt depth image to "padding" depth range type
        if "XYZ" in self.network_input:
            depth_img, joints_2D, trans_input, _, _ = augment_affine(depth_img, 
                                                      joints_2D, 
                                                      input_w=self.input_img_size[0], 
                                                      input_h=self.input_img_size[0], 
                                                      interpolation_method=cv2.INTER_NEAREST, 
                                                      aug_mode=self.aug_mode)
        
            # create visible depth map
            depth_img_vis = ((depth_img * 255) / depth_img.max()).astype(np.uint8)[..., None]
            depth_img_vis = np.concatenate([depth_img_vis, depth_img_vis, depth_img_vis], -1)
            t4 = time.time()

            # compute XYZ image
            xyz_img = depthmap2points(depth_img / 1000, fx=fx*scale, fy=fy*scale, cx=intrinsic[0, 2]*scale, cy=intrinsic[1, 2]*scale)
            xyz_img[..., 0] = xyz_img[..., 0] / 3.
            xyz_img[..., 1] = xyz_img[..., 1] / 2.
            xyz_img[..., 2] = xyz_img[..., 2] / 5.
            xyz_img[depth_img == 0] = 0

        
        elif "RGB" in self.network_input:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            rgb_img, joints_2D, trans_input, _, _ = augment_affine(rgb_img, 
                                                      joints_2D, 
                                                      input_w=self.input_img_size[0], 
                                                      input_h=self.input_img_size[0], 
                                                      interpolation_method=cv2.INTER_LINEAR, 
                                                      aug_mode=self.aug_mode,
                                                      mean=self.mean,
                                                      std=self.std,
                                                      )  
            rgb_img_vis = ((rgb_img * self.std) + self.mean ) * 255
            #print(self.aug_mode)
            
        #print("trans_input", trans_input)
        #print("intrinsic", intrinsic)
        intrinsic *= scale
        intrinsic[1, 2] += trans_input[1, 2]
        #print("intrinsic", intrinsic)
        #print("joints_2d", joints_2D)
        #print(rgb_img.shape)
        if self.cond_norm:
            input_joints_2D = np.zeros_like(joints_2D)
            input_joints_2D[:, 0] = (joints_2D[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0]
            input_joints_2D[:, 1] = (joints_2D[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1]
        else:
            input_joints_2D = np.array(joints_2D) / self.input_img_size[0] # map to [0, 1]

        R2C_Mat_after_aug = np.array(sample["R2C_Mat"]) # 3 x 3
        R2C_Trans_after_aug = np.array(sample["R2C_Trans"]) # 3
        R2C_Pose_after_aug = np.eye(4, 4)
        R2C_Pose_after_aug[:3, :3] = R2C_Mat_after_aug
        R2C_Pose_after_aug[:3, 3] = R2C_Trans_after_aug 
        output = {
            'intrinsic': intrinsic,
            'joints_2D_uv': input_joints_2D,
            "joints_2D_uv_raw" : np.array(joints_2D),
            'joints_3D_Z': joints_3D_Z,
            "joints_3D_kps" : torch.from_numpy(np.array(sample["joints_kps"]))[[0,2,3,4,6,7,8]],
            "R2C_Pose" : R2C_Pose_after_aug[:3, :],
            "rgb_path" : sample['rgb_file'],
            "joints_7" : torch.from_numpy(np.array(sample["joints_8"])[:7]).float(),
        }
        if self.network_task == '2d_RPE':
            output['joints_3D_depth'] = joints_3D_depth
            output['heatmap_depth'] = torch.from_numpy(heatmaps.astype(np.float32).transpose(2, 0, 1))
            if "RGB" in self.network_input:
                output['heatmap_rgb'] = torch.from_numpy(heatmaps.astype(np.float32).transpose(2, 0, 1))
        
        if "XYZ" in self.network_input:
            output['input_tensor'] = torch.from_numpy(xyz_img.transpose(2, 0, 1))
            output["vis"] = depth_img_vis
        if "RGB" in self.network_input:
            output["input_tensor"] = torch.from_numpy(rgb_img.transpose(2, 0, 1))
            output["vis"] = rgb_img_vis
        t7 = time.time()

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
        self.dataset_dict = {"train" : self.train_dataset_dir,
                        "val" : self.val_dataset_dir,
                        "real" : self.real_dataset_dir,}
        splits = ["train", "val", "real"]
        data = defaultdict(list)
        for split in splits:
            if split == "train" or split == "val":
                iter = 0
                dataset_dir = self.dataset_dict[split]
                depth_files = glob.glob(os.path.join(dataset_dir, "*", "*.npy"))
                depth_files.sort()
                
                depth_files = depth_files
                
                for depth_file in tqdm(depth_files, f"Loading {split} ..."):
                     # rgb_file like this : '/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/10443/0029_color.png'
                    scene_id, frame_name = depth_file.split('/')[-2], depth_file.split('/')[-1]
                    img_name = f"{scene_id}_{frame_name}"
                    
                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    #depth_file = rgb_file
                    rgb_file = depth_file.replace('simDepthImage.npy', 'color.png')
                    depth_pf_file = depth_file
                    mask_file = rgb_file.replace("color.png", "mask.exr")
                    joints_file = depth_file.replace('simDepthImage.npy', 'meta.json')
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
                            "mask_file" : mask_file,
                            "pred_mask_file" : mask_file, 
                            "depth_pf_file" : depth_pf_file,
                            'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]
                            "joints_8" : joints_8_pos, 
                            "joints_kps" : json_keypoints_pos,
                            "R2C_Mat" :  R2C_Mat,
                            "R2C_Trans" : R2C_Trans
                        }
    
                    data[split].append(sample)
            
            elif split == "real":
                dataset_dir = self.dataset_dict[split]
                rgb_files = glob.glob(os.path.join(dataset_dir, "*.rgb.jpg"))  
                 
                print("length of dataset_dir", len(rgb_files))
                rgb_files.sort() 
                
                for rgb_file in tqdm(rgb_files, f"Loading {split} ..."):
                    # rgb_file like this : '/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/10443/0029_color.png'
                    frame_name = (rgb_file.split('/')[-1]).replace(".rgb.jpg", "")
                    
                    # Here the dataset requires depth 8 and depth 16, so how to do that?
                    depth_file = rgb_file
                    meta_file = rgb_file.replace(".rgb.jpg", ".pkl") 
                    mask_file = rgb_file.replace(".rgb.jpg", "_mask.exr")
                    pred_mask_file = rgb_file.replace(".rgb.jpg", "_ours_mask.exr")
                    with open(meta_file, 'rb') as fd:
                        #json_data = json.load(fd)[0]
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
    
    
    
    
    
    
    
    
    
    
    
    