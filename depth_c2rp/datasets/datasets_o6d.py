from enum import IntEnum
import time
import albumentations as albu
import numpy as np
import os
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms
from depth_c2rp.utils.image_proc import crop_and_resize_img, crop_and_resize_mask, crop_and_resize_simdepth, normalize_image, res_crop_and_resize_simdepth, get_whole_simdepth
from depth_c2rp.utils.utils import load_keypoints_and_joints, matrix_to_quaternion, quaternionToRotation, load_camera_intrinsics, load_all

class Depth_dataset(TorchDataset):
    def __init__(
        self,
        seq_dataset,
        manipulator_name,
        keypoint_names,
        joint_names,
        input_resolution, # [height, width]
        mask_dict,
        camera_K=None,
        is_res=False,
        device="cpu",
        twonplus2=False,
        seq_frame=3,
        multi_frame=False,
        num_classes = 3
        ):
        self.dataset = seq_dataset
        self.manipulator_name = manipulator_name
        self.keypoint_names = keypoint_names
        self.joint_names = joint_names 
        self.camera_K = camera_K
        self.camera_K_tensor = torch.from_numpy(self.camera_K).float()
        self.is_res = is_res
        self.mask_dict = mask_dict
        self.device = device
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.input_resolution = input_resolution 
        self.seq_frame = seq_frame
        self.multi_frame = multi_frame
        self.num_classes = num_classes
        self.twonplus2 = twonplus2
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        start_time = time.time()
        datum = self.dataset[index]
        scene_id, frame_id = datum["next_frame_name"].split("/")
        frame_id = int(frame_id)
        
        if frame_id % self.seq_frame == 0:
            if self.multi_frame:
                next_frame_id = datum["prev_frame_name"]
                next_frame_img_path = datum["prev_frame_img_path"]
                next_frame_meta_path = datum["prev_frame_data_path"]
                next_frame_mask_path = datum["prev_frame_mask_path"]
                next_frame_simdepth_path = datum["prev_frame_simdepth_path"]
                old_id = str(frame_id).zfill(4)
                new_id = str(frame_id - self.seq_frame).zfill(4)
                prev_frame_id = os.path.join(scene_id, new_id)
                prev_frame_img_path = datum["next_frame_img_path"].replace(old_id + "_color.png", new_id + "_color.png")
                prev_frame_meta_path = datum["next_frame_data_path"].replace(old_id + "_meta.json", new_id + "_meta.json")
                prev_frame_mask_path = datum["next_frame_mask_path"].replace(old_id + "_mask.exr", new_id + "_mask.exr")
                prev_frame_simdepth_path = datum["next_frame_simdepth_path"].replace(old_id + "_simDepthImage.exr", new_id + "_simDepthImage.exr")
            else:
                # [2, 3] -> [0, 0]
                old_id = str(frame_id).zfill(4)
                new_id = str(frame_id - self.seq_frame).zfill(4)
                prev_frame_id = os.path.join(scene_id, new_id)
                prev_frame_img_path = datum["next_frame_img_path"].replace(old_id + "_color.png", new_id + "_color.png")
                prev_frame_meta_path = datum["next_frame_data_path"].replace(old_id + "_meta.json", new_id + "_meta.json")
                prev_frame_mask_path = datum["next_frame_mask_path"].replace(old_id + "_mask.exr", new_id + "_mask.exr")
                prev_frame_simdepth_path = datum["next_frame_simdepth_path"].replace(old_id + "_simDepthImage.exr", new_id + "_simDepthImage.exr")
                next_frame_id = os.path.join(scene_id, new_id)
                next_frame_img_path = datum["next_frame_img_path"].replace(old_id + "_color.png", new_id + "_color.png")
                next_frame_meta_path = datum["next_frame_data_path"].replace(old_id + "_meta.json", new_id + "_meta.json")
                next_frame_mask_path = datum["next_frame_mask_path"].replace(old_id + "_mask.exr", new_id + "_mask.exr")
                next_frame_simdepth_path = datum["next_frame_simdepth_path"].replace(old_id + "_simDepthImage.exr", new_id + "_simDepthImage.exr")   
        else:
            prev_frame_id = datum["prev_frame_name"]
            prev_frame_img_path = datum["prev_frame_img_path"]
            prev_frame_meta_path = datum["prev_frame_data_path"]
            prev_frame_mask_path = datum["prev_frame_mask_path"]
            prev_frame_simdepth_path = datum["prev_frame_simdepth_path"]
            next_frame_id = datum["next_frame_name"]
            next_frame_img_path = datum["next_frame_img_path"]
            next_frame_meta_path = datum["next_frame_data_path"]
            next_frame_mask_path = datum["next_frame_mask_path"]
            next_frame_simdepth_path = datum["next_frame_simdepth_path"]


        
        #prev_keypoints, prev_joints = load_keypoints_and_joints(prev_frame_meta_path, self.keypoint_names, self.joint_names)
        if not self.twonplus2:
            next_keypoints, next_joints = load_keypoints_and_joints(next_frame_meta_path, self.keypoint_names, self.joint_names)
        else: 
            next_keypoints, next_joints, next_joints_2nplus2 = load_all(next_frame_meta_path, self.keypoint_names, self.joint_names)
        
#        load_time = time.time()
#        print("load_time", load_time - start_time)
        
        
        # generate image and transform to network input resolution
        #prev_image = crop_and_resize_img(prev_frame_img_path, self.input_resolution) # ???resize???, hxwx3
        next_image, next_bbox = crop_and_resize_img(next_frame_img_path, self.input_resolution)
        assert next_image.shape[0] == self.input_resolution[0]
        assert next_image.shape[1] == self.input_resolution[1]
        #prev_image_np = normalize_image(prev_image, self.mean, self.std) # c x h x w
        next_image_np = normalize_image(next_image, self.mean, self.std)
        
        img_time = time.time()
#        print("img_time", img_time - load_time)
        
        # generate mask
        #prev_mask_np = crop_and_resize_mask(prev_frame_mask_path, self.input_resolution, self.mask_dict, self.num_classes) # num_classes x h x w
        next_mask_np = crop_and_resize_mask(next_frame_mask_path, self.input_resolution, self.mask_dict, self.num_classes)
        
        mask_time = time.time()
#        print("mask_time", mask_time-img_time)
        
        # generate depth
        #prev_simdepth_np, prev_xy_wrt_cam, prev_uv, prev_normals_crop = crop_and_resize_simdepth(prev_frame_simdepth_path, self.input_resolution, uv=True, xy=True, nrm=True) # 1 x h x w
        if not self.is_res:
            next_simdepth_np, next_xy_wrt_cam, next_uv, next_normals_crop = crop_and_resize_simdepth(next_frame_simdepth_path, \
                                                                        self.input_resolution, uv=True, xy=True, nrm=True,device=self.device)
        else:
            next_simdepth_np, next_xy_wrt_cam, next_uv, next_normals_crop, next_xyz_rp = res_crop_and_resize_simdepth(next_frame_simdepth_path, \
                                                                        self.input_resolution, uv=True, xy=True, nrm=True,device=self.device, camera_K=self.camera_K) 
        simdepth_time = time.time()
#        print("simdepth_time", simdepth_time - mask_time)
        
        # generate 3D information
        #prev_base_quaternion_np = np.array(prev_keypoints["O2C_wxyz"][0], dtype=np.float32) # (4, )
        next_base_quaternion_np = np.array(next_keypoints["O2C_wxyz"][0], dtype=np.float32)
        #prev_base_trans_np = np.array(prev_keypoints["Location_wrt_cam"][0], dtype=np.float32) # (3, )
        next_base_trans_np = np.array(next_keypoints["Location_wrt_cam"][0], dtype=np.float32)
        #prev_kps_wrt_cam_np = np.array(prev_keypoints["Location_wrt_cam"], dtype=np.float32) # (??????????????????)
        next_kps_wrt_cam_np = np.array(next_keypoints["Location_wrt_cam"], dtype=np.float32)
        #prev_joints_pos_np = np.array(prev_joints["Angle"], dtype=np.float32)[:, None] # (8,)
        next_joints_pos_np = np.array(next_joints["Angle"], dtype=np.float32)[:, None]
        #prev_joints_wrt_cam_np = np.array(prev_joints["Location_wrt_cam"], dtype=np.float32)
        next_joints_wrt_cam_np = np.array(next_joints["Location_wrt_cam"], dtype=np.float32)
        #prev_base_rot_np = np.array(quaternionToRotation(prev_base_quaternion_np)) # (3, 3)
        next_base_rot_np = np.array(quaternionToRotation(next_base_quaternion_np))
        next_bbox_np = np.array(next_bbox, dtype=np.float32) #(4, )
        
        #prev_joints_wrt_rob_np = (prev_base_rot_np.T @ ((prev_joints_wrt_cam_np - prev_base_trans_np[None, :]).T)).T
        next_joints_wrt_rob_np = (next_base_rot_np.T @ ((next_joints_wrt_cam_np - next_base_trans_np[None, :]).T)).T
        
        next_whole_simdepth_np = get_whole_simdepth(next_frame_simdepth_path)
        
        # Convert data to tensors -use float32 size 
        #prev_image_as_input_tensor = torch.from_numpy(prev_image_np).float()
        next_image_as_input_tensor = torch.from_numpy(next_image_np).float()
        #prev_mask_as_input_tensor = torch.from_numpy(prev_mask_np).long()
        next_mask_as_input_tensor = torch.from_numpy(next_mask_np).long()
        #prev_simdepth_as_input_tensor = torch.from_numpy(prev_simdepth_np).float()
        next_simdepth_as_input_tensor = torch.from_numpy(next_simdepth_np).float()
        next_whole_simdepth_as_input_tensor = torch.from_numpy(next_whole_simdepth_np).float()
        #prev_base_quaternion_tensor = torch.from_numpy(prev_base_quaternion_np).float()
        next_base_quaternion_tensor = torch.from_numpy(next_base_quaternion_np).float()
        #prev_base_trans_tensor = torch.from_numpy(prev_base_trans_np).float()
        next_base_trans_tensor = torch.from_numpy(next_base_trans_np).float()
        #prev_kps_wrt_cam_tensor = torch.from_numpy(prev_kps_wrt_cam_np).float()
        next_kps_wrt_cam_tensor = torch.from_numpy(next_kps_wrt_cam_np).float()
        #prev_joints_pos_tensor = torch.from_numpy(prev_joints_pos_np).float()
        next_joints_pos_tensor = torch.from_numpy(next_joints_pos_np).float()
        #prev_joints_wrt_cam_tensor = torch.from_numpy(prev_joints_wrt_cam_np).float()
        next_joints_wrt_cam_tensor = torch.from_numpy(next_joints_wrt_cam_np).float()
        #prev_joints_wrt_rob_tensor = torch.from_numpy(prev_joints_wrt_rob_np).float()
        next_joints_wrt_rob_tensor = torch.from_numpy(next_joints_wrt_rob_np).float()
        next_bbox_tensor = torch.from_numpy(next_bbox_np).float()
        #if prev_xy_wrt_cam is not None and next_xy_wrt_cam is not None:
        if  next_xy_wrt_cam is not None:
            #prev_xy_wrt_cam_tensor = torch.from_numpy(prev_xy_wrt_cam).float()
            next_xy_wrt_cam_tensor = torch.from_numpy(next_xy_wrt_cam).float()
        else:
            #prev_xy_wrt_cam_tensor = None
            next_xy_wrt_cam_tensor = None
        #if prev_uv is not None and next_uv is not None:
        if next_uv is not None:
            #prev_uv_tensor = torch.from_numpy(prev_uv).float()
            next_uv_tensor = torch.from_numpy(next_uv).float()
        else:  
            #prev_uv_tensor = None
            next_uv_tensor = None
        #if prev_normals_crop is not None and next_normals_crop is not None:
        if next_normals_crop is not None:
            #prev_normals_crop_tensor = torch.from_numpy(prev_normals_crop).float()
            next_normals_crop_tensor = torch.from_numpy(next_normals_crop).float()
        else:
            #prev_normals_crop_tensor = None
            next_normals_crop_tensor = None
        
        transform_time = time.time()
#        print("transform_time", transform_time - simdepth_time)
        
        # sample
        sample = {
#                  "prev_frame_img_path" :  prev_frame_img_path,
#                  "prev_frame_img_as_input" : prev_image_as_input_tensor,
#                  "prev_frame_mask_as_input" : prev_mask_as_input_tensor,
#                  "prev_frame_simdepth_as_input" : prev_simdepth_as_input_tensor,
#                  "prev_frame_base_quaternion" : prev_base_quaternion_tensor,
#                  "prev_frame_base_trans" : prev_base_trans_tensor,
#                  "prev_frame_kps_wrt_cam" : prev_kps_wrt_cam_tensor,
#                  "prev_frame_joints_pos" : prev_joints_pos_tensor,
#                  "prev_frame_joints_wrt_cam" : prev_joints_wrt_cam_tensor,
#                  "prev_frame_joints_wrt_rob" : prev_joints_wrt_rob_tensor,
#                  "prev_frame_xy_wrt_cam" : prev_xy_wrt_cam_tensor,
#                  "prev_frame_uv" : prev_uv_tensor,
#                  "prev_normals_crop" : prev_normals_crop_tensor,
                  "next_frame_img_path" :  next_frame_img_path,
                  "next_frame_img_as_input" : next_image_as_input_tensor,
                  "next_frame_mask_as_input" : next_mask_as_input_tensor,
                  "next_frame_simdepth_as_input" : next_simdepth_as_input_tensor,
                  "next_frame_whole_simdepth_as_input" : next_whole_simdepth_as_input_tensor,
                  "next_frame_base_quaternion" : next_base_quaternion_tensor,
                  "next_frame_base_trans" : next_base_trans_tensor,
                  "next_frame_kps_wrt_cam" : next_kps_wrt_cam_tensor,
                  "next_frame_joints_pos" : next_joints_pos_tensor,
                  "next_frame_joints_wrt_cam" : next_joints_wrt_cam_tensor,
                  "next_frame_joints_wrt_rob" : next_joints_wrt_rob_tensor,
                  "next_frame_xy_wrt_cam" : next_xy_wrt_cam_tensor,
                  "next_frame_uv" : next_uv_tensor,
                  "next_normals_crop" : next_normals_crop_tensor,
                  "next_camera_K" : self.camera_K_tensor,
                  "next_bbox" : next_bbox_tensor
                  }
        if self.is_res:
            next_xyz_rp_tensor = torch.from_numpy(next_xyz_rp).float()
            sample.update({"next_xyz_rp" : next_xyz_rp_tensor})
        else:
            next_xyz_rp_tensor = torch.zeros(1, 3).float()
            sample.update({"next_xyz_rp" : next_xyz_rp_tensor})
        
        if self.twonplus2:
            next_joints_2nplus2_wrt_cam_np = np.array(next_joints_2nplus2["Location_wrt_cam"], dtype=np.float32) # 16
            next_joints_2nplus2_wrt_cam_tensor = torch.from_numpy(next_joints_2nplus2_wrt_cam_np).float()
            sample.update({"next_joints_2nplus2" : next_joints_2nplus2_wrt_cam_tensor})
        else:
            sample.update({"next_joints_2nplus2" : next_joints_wrt_cam_tensor})
            
        end_time = time.time()
#        print('all_time', end_time - start_time)
        return sample

if __name__ == "__main__":
  a()