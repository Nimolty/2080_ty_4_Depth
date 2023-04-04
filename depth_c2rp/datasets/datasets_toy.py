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
import os
import copy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyquaternion import Quaternion
import glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

#from utils.data_augmentation import augment_3d
#from utils.depth_utils import depthmap2pointcloud
#from utils.depth_utils import depthmap2points
#from utils.depth_utils import overlay_points_on_image, mosaic_images
#from utils.preprocessing import apply_depth_normalization_16bit_image
#from utils.utils import heatmap_from_kpoints_array, gkern
from depth_c2rp.utils.spdh_utils import augment_3d, depthmap2pointcloud, depthmap2points, overlay_points_on_image, mosaic_images
from depth_c2rp.utils.spdh_utils import apply_depth_normalization_16bit_image, heatmap_from_kpoints_array, gkern

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
                 aug_mode: bool = True, noise: bool = False, demo: bool = False):
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
        self.data = self.load_data()

    def __len__(self):
        return len(self.data[self.mode])

    def __getitem__(self, idx):    
        sample = self.data[self.mode][idx].copy()
        # image loading (depth and/or RGB)
        joints_3D_Z = np.zeros((sample['joints'].shape[0], 3))
        #print("sample_joints_3d", sample["joints"]) 
        for n, joint in enumerate(sample["joints"]):
            point3d = np.asarray([joint[0], joint[1], joint[2]]) # right handed reference frame
            joints_3D_Z[n] = point3d.copy()
            
        output = {'joints_3D_Z': joints_3D_Z,}
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
                joints_file = rgb_file.replace('color.png', 'meta.json')
                with open(joints_file, 'r') as fd:
                    json_data = json.load(fd)[0]
                json_keypoints_data = json_data["keypoints"]
                #json_joints_data = json_data["joints"]
                json_joints_data = json_data["joints_3n"]
                
                keypoints_r2c_data = [json_keypoints_data[idx]["R2C_mat"] for idx in range(len(json_keypoints_data))]
                joints_loc_wrt_cam_data = [json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))]
                joints_pos = np.zeros((len(self.JOINT_NAMES), 3), dtype=np.float32)
                for idx, k in enumerate(self.JOINT_NAMES):
#                    joint_info = JOINT_INFOS[k]
#                    kp_id = joint_info["base"]
#                    r2c_mat = keypoints_r2c_data[kp_id]
#                    r2c_mat_np = np.asarray(r2c_mat)
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
                        'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]
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
    
    
    
    
    
    
    
    
    
    
    
    
