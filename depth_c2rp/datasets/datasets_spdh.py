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

from depth_c2rp.utils.spdh_utils import augment_3d, depthmap2pointcloud, depthmap2points, overlay_points_on_image, mosaic_images
from depth_c2rp.utils.spdh_utils import apply_depth_normalization_16bit_image, heatmap_from_kpoints_array, gkern, compute_rigid_transform

#JOINTS_NAMES = [
#    'head',
#    'right_upper_shoulder',
#    'right_lower_shoulder',
#    'right_upper_elbow',
#    'right_lower_elbow',
#    'right_upper_forearm',
#    'right_lower_forearm',
#    'right_wrist',
#    'left_upper_shoulder',
#    'left_lower_shoulder',
#    'left_upper_elbow',
#    'left_lower_elbow',
#    'left_upper_forearm',
#    'left_lower_forearm',
#    'left_wrist',
#    'base',
#    # 'r_gripper_r_finger',
#    # 'r_gripper_l_finger',
#    # 'l_gripper_r_finger',
#    # 'l_gripper_l_finger',
#]
JOINT_NAMES = ["panda_joint1",
               "panda_joint2",
               "panda_joint3",
               "panda_joint4",
               "panda_joint5",
               "panda_joint6",
               "panda_joint7",
               "panda_finger_joint1"
              ]
#JOINT_INFOS = 
#              {
#              "panda_joint1" : {"index" : 2, "direction" : 1, "base" : 0, "offset" : [0.0, 0.0, 0.14]},
#              "panda_joint2" : {"index" : 1, "direction" : 1, "base" : 1, "offset" : [0.0, 0.0, 0.0]},
#              "panda_joint3" : {"index" : 1, "direction" : -1, "base" : 3, "offset" : [0.0, 0.0, -0.1210]},
#              "panda_joint4" : {"index" : 1, "direction" : -1, "base" : 4, "offset" : [0.0, 0.0, 0.0]},
#              "panda_joint5" : {"index" : 1, "direction" : 1, "base" : 5, "offset" : [0.0, 0.0, -0.2590]},
#              "panda_joint6" : {"index" : 1, "direction" : -1, "base" :5, "offset" : [0.0, 0.0158, 0.0]},
#              "panda_joint7" : {"index" : 1, "direction" : -1, "base" : 7, "offset" : [0.0, 0.0, 0.0520]},
#              "panda_finger_joint1" : {"base" : 8, "offset" : [0.0, 0.0, 0.0584]}
#              }
#
#KEYPOINTS_NAMES = ["Link0", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "panda_hand", "panda_finger_joint1", "panda_finger_joint2"]

def init_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


class Depth_dataset(Dataset):
    def __init__(self, train_dataset_dir: Path, val_dataset_dir : Path, joint_names: list, run: list, init_mode: str = 'train', img_type: str = 'D',
                 raw_img_size: tuple = (640, 360), input_img_size: tuple = (384, 216), sigma: int = 4., norm_type: str = 'mean_std',
                 network_input: str = 'D', network_output: str = 'H', network_task: str = '3d_RPE',
                 depth_range: tuple = (500, 3380, 15), depth_range_type: str = 'normal', aug_type: str = '3d',
                 aug_mode: bool = True, noise: bool = False, demo: bool = False, load_mask: bool = False, mask_dict : dict = {}, unnorm_depth: bool=False):
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
        self.JOINT_NAMES = joint_names
        #self.JOINT_INFOS = joint_infos
        #self.KEYPOINTS_NAMES = keypoints_names
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
        self.data = self.load_data()

    def __len__(self):
        return len(self.data[self.mode])

    def __getitem__(self, idx):    
        t1 = time.time()
        sample = self.data[self.mode][idx].copy()
        # image loading (depth and/or RGB)
        depth_img = cv2.imread(sample['depth_file'], cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32) * 1000 # milimeters 
        depth_img = cv2.resize(depth_img, tuple(self.input_img_size), interpolation=cv2.INTER_NEAREST)
        
        depth_pf_img = cv2.imread(sample['depth_pf_file'], cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32) * 1000 # milimeters 
        depth_pf_img = cv2.resize(depth_pf_img, tuple(self.input_img_size), interpolation=cv2.INTER_NEAREST)
        
        H, W = depth_img.shape
        assert H == self.input_img_size[1]
        assert W == self.input_img_size[0]
        # RGB and depth scale
        scale_x, scale_y = depth_img.shape[1] / self.raw_img_size[0], depth_img.shape[0] / self.raw_img_size[1]
        
        # image size divided by 32 should be an even value (for SH network)
        depth_img = depth_img[12:-12, :] # 192 * 384
        depth_pf_img = depth_pf_img[12:-12, :]
        if self.load_mask:
            mask_file = cv2.imread(sample["mask_file"], cv2.IMREAD_UNCHANGED)[:, :, 2]
            mask_file_res = np.zeros_like(mask_file)
            assert self.mask_dict != {}
            for idx, (key, value) in enumerate(self.mask_dict.items()):
                mask_file_res[np.where(mask_file == value)] = idx+1
            
            #mask_file_res[np.where(mask_file != 1)] = 1
            mask_file_res = cv2.resize(mask_file_res[:, :, None], (W, H), interpolation=cv2.INTER_NEAREST)
            mask_file_res = mask_file_res[12:-12, :][:, :, None]
            #print(mask_file_res.shape)
        
        # adapt depth image to "padding" depth range type
        if self.depth_range_type == 'padding':
            Z_min, Z_max, dZ = self.depth_range
            new_img_h = (Z_max - Z_min) // dZ
            padding = int(np.abs(depth_img.shape[0] - new_img_h) // 2)
            depth_img = cv2.copyMakeBorder(depth_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)
            depth_pf_img = cv2.copyMakeBorder(depth_pf_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)
        
        # copy unnormed depth img
        if self.unnorm_depth:
            depth_img_unnorm = depth_img.copy() / 1000
            depth_pf_img_unnorm = depth_pf_img.copy() / 1000
        
        # initialize intrinsic matrix
        cam_settings_data = json.load(open(str(self.train_dataset_dir / "_camera_settings.json"), 'r'))
        fx, fy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fx"]* scale_x, cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fy"]* scale_y
        cx, cy = depth_img.shape[1] / 2, depth_img.shape[0] / 2
        intrinsic = np.asarray([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        
        t2 = time.time()
        
        joints_2D = np.zeros((sample['joints'].shape[0], 2))
        joints_3D_Z = np.zeros((sample['joints'].shape[0], 3))
        #print("sample_joints_3d", sample["joints"]) 
        for n, joint in enumerate(sample["joints"]):
            point3d = np.asarray([joint[0], joint[1], joint[2]]) # right handed reference frame
            joints_3D_Z[n] = point3d.copy()
            
            # I don't known whether to invert Y axis direction for 2D reprojection, so I just comment it on
            #point3[1] *= -1
            u, v, w = (intrinsic @ point3d[:, None]).T.squeeze(0)
            u = u / w
            v = v / w
            joints_2D[n] = [u, v]
        joints_3D_Z_copy = joints_3D_Z.copy()
        # load R2C Mat and Trans
        R2C_Mat_before_aug = np.array(sample["R2C_Mat"]) # 3 x 3
        R2C_Trans_before_aug = np.array(sample["R2C_Trans"]) # 3
        
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
                #point3d[..., 1] *= -1  # I dont' known whether to invert Y axis direction for 2D reprojection
                joints_2D = (intrinsic @ point3d[..., None]).squeeze(-1)
                joints_2D = (joints_2D / joints_2D[:, -1:])[:, :2]
        else:
            R2C_Mat_after_aug = R2C_Mat_before_aug
            R2C_Trans_after_aug = R2C_Trans_before_aug
        # create visible depth map
        depth_img_vis = ((depth_img * 255) / depth_img.max()).astype(np.uint8)[..., None]
        depth_img_vis = np.concatenate([depth_img_vis, depth_img_vis, depth_img_vis], -1)
        
        R2C_Pose_after_aug = np.eye(4, 4)
        R2C_Pose_after_aug[:3, :3] = R2C_Mat_after_aug
        R2C_Pose_after_aug[:3, 3] = R2C_Trans_after_aug 
        t3 = time.time()
#        joints_1d_gt = torch.from_numpy(np.array(sample["joints_8"]))[None, :, None].float()
#        joints_3D_Z_rob = compute_3n_loss(joints_1d_gt)
#        print("ours_computed_rot", R2C_Mat_after_aug)
#        print("ours_computed_trans", R2C_Trans_after_aug)
#        
#        KO_pose = compute_rigid_transform(joints_3D_Z_rob.float(), torch.from_numpy(joints_3D_Z_copy)[None, :, :].float())
#        print("KO_pose", KO_pose)
        
        '''
        DEBUG = False
        if DEBUG:
            import matplotlib.pyplot as plt
            depth16_img_vis_copy = depth16_img_vis.copy()
            if "RGB" in self.img_type:
                rgb_img_copy = rgb_img.copy()
                for n, joint_2D in enumerate(joints_2D):
                    print(JOINTS_NAMES[n])
                    cv2.circle(rgb_img_copy, (int(joint_2D[0]), int(joint_2D[1])), 10, (255, 0, 0), -1)
                plt.imshow(rgb_img_copy[..., ::-1].astype(np.uint8))
                plt.show()
            for n, joint_2D in enumerate(joints_2D):
                print(JOINTS_NAMES[n])
                cv2.circle(depth16_img_vis_copy, (int(joint_2D[0]), int(joint_2D[1])), 2, (255, 0, 0), -1)
            plt.imshow(depth16_img_vis_copy[..., ::-1].astype(np.uint8))
            plt.show()
        '''

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
        
        t4 = time.time()
        # compute XYZ image
        if "XYZ" in self.network_input:
            xyz_img = depthmap2points(depth_img / 1000, fx=fx, fy=fy, cx=intrinsic[0, 2], cy=intrinsic[1, 2])
            xyz_img[..., 0] = xyz_img[..., 0] / 3.
            xyz_img[..., 1] = xyz_img[..., 1] / 2.
            xyz_img[..., 2] = xyz_img[..., 2] / 5.
            xyz_img[depth_img == 0] = 0

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
            for n, (p, P) in enumerate(zip(joints_2D, joints_3D_Z)):
                P = P.copy() * 1000
                # compute distances (px) from point
                dst = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2)
                # convert sigma from mm to pixel
                sigma_pixel = sigma_mm * intrinsic[0, 0] / P[2]
                # compute the heatmap
                mu = 0.
                heatmaps_uv[n] = np.exp(-((dst - mu) ** 2 / (2.0 * sigma_pixel ** 2)))

            # UZ heatmaps
            z = np.arange(Z_min, Z_max, dZ)
            heatmaps_uz = np.zeros((joints_2D.shape[0],depth_img.shape[0], depth_img.shape[1])) # Is there any bugs in this line???
            for n, P in enumerate(joints_3D_Z):
                P = P.copy() * 1000
                # compute X for each value of x (u) at each slice z
                x = np.arange(depth_img.shape[1])  # [u]
                x_unsqueezed = x[None, :]  # [1, u]
                z = np.arange(Z_min, Z_max, dZ)  # [Z / dZ]
                z_unsqueezed = z[:, None]  # [Z / dZ, 1]
                fx, _, cx = intrinsic[0]
                X = (x_unsqueezed - cx) * z_unsqueezed / fx  # [Z / dZ, u]
                # compute distances (mm) from point
                dst_mm = np.sqrt((X - P[0]) ** 2 + (z_unsqueezed - P[2]) ** 2)  # [Z / dZ, u]
                # compute heatmap
                heatmaps_uz[n] = np.exp(-(dst_mm ** 2 / (2.0 * sigma_mm ** 2)))  # [Z / dZ, u]
            
            heatmaps_25d = np.concatenate((heatmaps_uv, heatmaps_uz), axis=0)

        # keypoint to heatmap transform
        if self.network_task == '2d_RPE':
            heatmaps = heatmap_from_kpoints_array(kpoints_array=joints_2D, shape=depth_img.shape[:2],
                                                  sigma=self.sigma)

        # mean and std for 2D and 3D joints
        #stats = np.atleast_1d(np.load(str(self.dataset_dir / "mean_std_stats.npy"), allow_pickle=True))[0]
        #print(joints_3D_Z.cuda())
        
        t5 = time.time()
#        print("t5 - t4", t5 -t4)
#        print("t4 - t3", t4 -t3)
#        print("t3 - t2", t3 -t2)
#        print("t2 - t1", t2 -t1)
        
        output = {
            'depth': torch.from_numpy(depth_img[None, ...]),
            'depthvis': depth_img_vis,
            'z_values': z_values,
            'K_depth': intrinsic,
            'joints_2D_depth': joints_2D,
            'joints_3D_Z': joints_3D_Z,
            "joints_3D_kps" : torch.from_numpy(np.array(sample["joints_kps"]))[[0,2,3,4,6,7,8]],
            "rgb_path" : sample['rgb_file'],
            "joints_7" : torch.from_numpy(np.array(sample["joints_8"])[:7]).float(),
            "R2C_Pose" : R2C_Pose_after_aug[:3, :]#torch.from_numpy(R2C_Pose_after_aug)[:3, :].float(), 
            
        }
        if self.network_task == '2d_RPE':
            output['joints_3D_depth'] = joints_3D_depth
            output['heatmap_depth'] = torch.from_numpy(heatmaps.astype(np.float32).transpose(2, 0, 1))
            if "RGB" in self.network_input:
                output['heatmap_rgb'] = torch.from_numpy(heatmaps.astype(np.float32).transpose(2, 0, 1))
        else:
            output['heatmap_25d'] = torch.from_numpy(heatmaps_25d.astype(np.float32))
        if "XYZ" in self.network_input:
            output['xyz_img'] = torch.from_numpy(xyz_img.transpose(2, 0, 1))
        if self.load_mask:
            output["mask"] = torch.from_numpy(mask_file_res).float()
        if self.unnorm_depth:
            output["unnorm_depth"] = torch.from_numpy(depth_img_unnorm[None, ...]).float()
            output["unnorm_pf_depth"] = torch.from_numpy(depth_pf_img_unnorm[None, ...]).float()

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

    @property
    def mode(self):
        return self._mode 

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'val', 'test']
        self._mode = value

    def load_data(self):
        dataset_dict = {"train" : self.train_dataset_dir,
                        "val" : self.val_dataset_dir}
        splits = ["train", "val"]
        data = defaultdict(list)
        for split in splits:
            iter = 0
            dataset_dir = dataset_dict[split]
            rgb_files = glob.glob(os.path.join(dataset_dir, "*", "*_color.png"))
            rgb_files.sort()
            for rgb_file in tqdm(rgb_files, f"Loading {split} ..."):
                 # rgb_file like this : '/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/10443/0029_color.png'
                scene_id, frame_name = rgb_file.split('/')[-2], rgb_file.split('/')[-1]
                img_name = f"{scene_id}_{frame_name}"
                
                # Here the dataset requires depth 8 and depth 16, so how to do that?
                depth_file = rgb_file.replace('color.png', 'simDepthImage.exr')
                depth_pf_file = rgb_file.replace("color.png", "depth_60.exr")
                joints_file = rgb_file.replace('color.png', 'meta.json')
                mask_file = rgb_file.replace("color.png", "mask.exr")
                with open(joints_file, 'r') as fd:
                    json_data = json.load(fd)[0]
                json_keypoints_data = json_data["keypoints"]
                #json_joints_data = json_data["joints"]
                #json_joints_data = json_data["joints_3n_fixed_40"]
                
                json_joints_data = json_data["joints_3n_fixed_42"]
                
                json_joints_8_data = json_data["joints"]
                joints_8_pos = [kp["position"] for idx, kp in enumerate(json_joints_8_data)] 
                json_keypoints_pos = [kp["location_wrt_cam"] for idx, kp in enumerate(json_keypoints_data)][:-2]
                
                R2C_Mat = json_keypoints_data[0]["R2C_mat"]
                R2C_Trans = json_keypoints_data[0]["location_wrt_cam"]
                joints_loc_wrt_cam_data = [json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))]
                
                #joints_loc_wrt_cam_data.append(json_keypoints_data[-1]["location_wrt_cam"])
                assert len(self.JOINT_NAMES) == len(joints_loc_wrt_cam_data)
                
                
                
                joints_pos = np.zeros((len(self.JOINT_NAMES), 3), dtype=np.float32)
                for idx, k in enumerate(self.JOINT_NAMES):
                    loc_wrt_cam = joints_loc_wrt_cam_data[idx]
#                    q = Quaternion(matrix=r2c_mat_np)
                    joints_pos[idx] = [loc_wrt_cam[0], 
                                       loc_wrt_cam[1],
                                       loc_wrt_cam[2],] 
#                                       q.x,
#                                       q.y,
#                                       q.z,
#                                       q.w]

                iter += 1

                sample = {
                        'rgb_file': rgb_file,
                        "depth_file" : depth_file, # mm
                        "depth_pf_file" : depth_pf_file,
                        "mask_file"  : mask_file,
                        'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]
                        "joints_8" : joints_8_pos, 
                        "joints_kps" : json_keypoints_pos,
                        "R2C_Mat" :  R2C_Mat,
                        "R2C_Trans" : R2C_Trans
                    }

                data[split].append(sample)
                    
                    

        return data

if __name__ == "__main__":
    PandaDataset = Depth_dataset(train_dataset_dir=Path("/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/"),
                                            val_dataset_dir=Path("/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_test_syn/"),
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
                                            aug_mode=False)
                                       #demo=self.cfg['demo'])
    a = dict()
    a["train"] = copy.copy(PandaDataset)
    PandaDataset.eval()
    a["val"] = PandaDataset
    for key, value in a.items():
        print(len(value))
    print("len(PandaDataset)", len(PandaDataset))
    for idx, item in enumerate(PandaDataset):
        print("item.keys()", item.keys())
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
    
    
    
    
    
    
    
    
    
    
    
    
