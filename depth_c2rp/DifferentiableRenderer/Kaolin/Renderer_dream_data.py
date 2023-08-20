import kaolin as kal
from kaolin.metrics.pointcloud import chamfer_distance

import torch
import torch.nn as nn
import math
from matplotlib import pyplot as plt
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import time
import random
from PIL import Image as PILImage
import json
import pickle5 as pickle
import glob
from depth_c2rp.DifferentiableRenderer.Kaolin.Render_utils import projectiveprojection_real, euler_angles_to_matrix, load_part_mesh, concat_part_mesh, exists_or_mkdir
from depth_c2rp.DifferentiableRenderer.Kaolin.Render_utils import quaternion_to_matrix, matrix_to_quaternion, euler_angles_to_matrix, matrix_to_euler_angles, seg_and_transform, compute_rotation_matrix_from_ortho6d
from depth_c2rp.DifferentiableRenderer.Kaolin.Render_utils import depthmap2pointcloud, overlay_points_on_image, mosaic_images, load_camera_intrinsics, load_dream_data

from tqdm import tqdm
import argparse

class DiffPFDepthRenderer():
   def __init__(self, cfg, device):
       super().__init__()
       """
       expect cfg is a dict
       """
       self.device = device
       self.CAD_model_paths = cfg["DR"]["CAD_MODEL_PATHS"]
       self.RT_lr = cfg["DR"]["RT_LR"]
       self.GA_lr = cfg["DR"]["GA_LR"]
       self.loss_fn = nn.L1Loss(reduction="sum")
       self.loss_chamfer = chamfer_distance
       #self.basis_change = torch.tensor([[[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]]],device=self.device)
   
   def load_mesh(self):
       self.vertices_list, self.faces_list, self.normals_list, self.faces_num_list = load_part_mesh(self.CAD_model_paths, self.device)
       self.single_face_indexs = torch.from_numpy(np.array(self.faces_num_list)).reshape(1, -1, 3, 3).float().to(self.device).contiguous()
       self.single_basis_change = torch.tensor([[[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]]],device=self.device)
       #self.face_indexs = self.face_indexs.repeat(bs, 1, 1, 1) # B x num_faces x 3 x 3
       self.single_ori_vertices = torch.cat(self.vertices_list,dim=1)

   
   def batch_mesh(self, bs):
       self.ori_vertices = self.single_ori_vertices.repeat(bs, 1, 1)
       self.face_indexs = self.single_face_indexs.repeat(bs, 1, 1, 1)
       self.basis_change = self.single_basis_change.repeat(bs, 1, 1)
      
   
   def concat_mesh(self, rot_type="quaternion"):
       self.joints_pos = torch.cat([self.joint1, self.joint2, self.joint3, self.joint4, self.joint5, self.joint6, self.joint7], dim=1)
   
       self.vertices_list_, self.R2C_list, self.T_list = concat_part_mesh(self.vertices_list, self.joints_pos, self.device,False)
       self.vertices = torch.cat(self.vertices_list_,dim=1)
       self.faces = torch.cat(self.faces_list,dim=0)
       
       if rot_type == "quaternion":
           self.quaternion_norm = self.quaternion / torch.norm(self.quaternion, dim=-1,keepdim=True)
           self.Rot_matrix = quaternion_to_matrix(self.quaternion_norm)
           #print(self.Rot_matrix)
       elif rot_type == "o6d":
           self.Rot_matrix = compute_rotation_matrix_from_ortho6d(self.o6dposes)
       else:
           raise ValueError

       self.vertices_camera = (torch.bmm(self.basis_change, (torch.bmm(self.Rot_matrix, self.vertices.permute(0,2,1)) + self.translation[:, :, None]))).permute(0,2,1)

       B, p, C = self.vertices_camera.shape
       self.vertices_ = torch.ones(B, p, C+1).to(self.device)
       self.vertices_[:, :, :C] = self.vertices_camera
       self.vertices_ndc_ = torch.matmul(self.vertices_, self.proj_T)
       self.vertices_ndc = self.vertices_ndc_ / self.vertices_ndc_[:, :, 2:3]
       self.face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(self.vertices_camera, self.faces)
       self.face_vertices_image = kal.ops.mesh.index_vertices_by_faces(self.vertices_ndc[..., :2], self.faces)
       self.face_vertices = kal.ops.mesh.index_vertices_by_faces(self.ori_vertices, self.faces)
       
#        print("self.face_vertices.shape", self.face_vertices.shape)
   
   def Rasterize(self):
       image_features, face_idx = kal.render.mesh.rasterize(
           self.depth_height, self.depth_width, self.face_vertices_camera[..., -1].contiguous(), self.face_vertices_image.contiguous(),
           [self.face_vertices, self.face_indexs], backend='nvdiffrast'
           ) 
       im_features, self.render_depth_mask = image_features
       self.render_depth_image = seg_and_transform(im_features, self.render_depth_mask, \
                                                self.Rot_matrix,self.basis_change,self.translation, self.device,self.R2C_list,self.T_list)
       return self.render_depth_image, self.render_depth_mask
   
   def loss_forward(self, input_depth, input_mask,img_path=None,update_idx=None, link_idx="whole", dr_order="single"):   
       if link_idx == "whole":     
           valid_mask = (input_mask > 1e-3)
           depth_valid_render_mask = (self.render_depth_mask > 1e-3)[:, 12:-12, :, :] # B x H x W x 3
       elif isinstance(link_idx, int) and dr_order == "single":
           valid_mask = (torch.abs(input_mask - link_idx-1) < 1e-3)
           depth_valid_render_mask = (torch.abs(self.render_depth_mask - link_idx-1) < 1e-3)[:, 12:-12, :, :] # B x H x W x 3
       elif isinstance(link_idx, int) and dr_order == "sequence":
           valid_mask = ((input_mask - link_idx-1) < 1e-3)
           valid_mask = valid_mask * ((input_mask) > 1e-3)
           depth_valid_render_mask = ((self.render_depth_mask - link_idx-1) < 1e-3)
           depth_valid_render_mask = (depth_valid_render_mask * ((self.render_depth_mask) > 1e-3))[:, 12:-12, :, :] # B x H x W x 3
       else:
           raise ValueError
               
       #test_mask = valid_mask * depth_valid_render_mask
       test_mask = depth_valid_render_mask
       
       all_depth = self.render_depth_image[:, 12:-12, :, :] * test_mask
       all_input = input_depth * test_mask
       render_mask_max = torch.max(((-1) * all_depth).flatten(1), dim=-1,keepdim=True)[0]
       input_max_mask = (all_input < render_mask_max[:, None, None, :] + 0.05)
       all_depth = all_depth * input_max_mask
       all_input = all_input * input_max_mask
       
       res = np.zeros_like(self.render_depth_image[:, 12:-12, :, :].detach().cpu().numpy())   
       for b in range(self.render_depth_image.shape[0]):
           render_mask_image = PILImage.fromarray((depth_valid_render_mask[b].detach().cpu().numpy()).astype(np.uint8) * 255)
           gt_rgb_image = cv2.imread(img_path[b])
           gt_rgb_image = cv2.cvtColor(gt_rgb_image, cv2.COLOR_RGB2BGR)
           gt_rgb_image = cv2.resize(gt_rgb_image, (depth_valid_render_mask.shape[2], self.render_depth_mask.shape[1]), interpolation=cv2.INTER_CUBIC)[12:-12, :, :]
            
           gt_rgb_image = PILImage.fromarray(gt_rgb_image.astype('uint8')).convert('RGB')
           blend_image = PILImage.blend(render_mask_image, gt_rgb_image, 0.7)
           #blend_image = PILImage.blend(render_mask_image, render_mask_image, 0.7) 
           blend_image_np = np.array(blend_image)
           res[b] = blend_image_np
            
       self.loss = self.loss_fn(all_depth[:, :, :, 2:3].float() * (-1.0), all_input.float()) / (test_mask.sum())
       print(self.loss.data)  
       return res
       
   def loss_forward_chamfer(self, input_xyz, input_mask,img_path=None,update_idx=None, link_idx="whole", dr_order="single"):   
       if link_idx == "whole":     
           valid_mask = (input_mask > 1e-3)
           depth_valid_render_mask = (self.render_depth_mask > 1e-3)[:, 12:-12, :, :] # B x H x W x 3
       elif isinstance(link_idx, int) and dr_order == "single":
           valid_mask = (torch.abs(input_mask - link_idx-1) < 1e-3)
           depth_valid_render_mask = (torch.abs(self.render_depth_mask - link_idx-1) < 1e-3)[:, 12:-12, :, :] # B x H x W x 3
       elif isinstance(link_idx, int) and dr_order == "sequence":
           valid_mask = ((input_mask - link_idx-1) < 1e-3)
           depth_valid_render_mask = ((self.render_depth_mask - link_idx-1) < 1e-3)[:, 12:-12, :, :] # B x H x W x 3
       else:
           raise ValueError
       
       res = np.zeros_like(self.render_depth_image[:, 12:-12, :, :].detach().cpu().numpy())
       self.loss = 0.0
       
       #test_mask = depth_valid_render_mask * valid_mask
       test_mask = depth_valid_render_mask
       
       for b in range(input_xyz.shape[0]):
           this_render_xyz = self.render_depth_image[b:b+1, 12:-12, :, :]
           this_input_xyz = input_xyz[b:b+1, :, :, :] # H x W x 3
           render_mask_max = torch.max(((-1) * this_render_xyz[:, :, :, -1]).flatten(1), dim=-1,keepdim=True)[0]
           input_max_mask = (this_input_xyz[:, :, :, -1] < render_mask_max[:, None, None, :] + 0.05)
          
           
           this_mask = test_mask[b:b+1, :, :, -1] * (input_max_mask.squeeze(1))
           
           this_render_xyz = this_render_xyz[this_mask].unsqueeze(0) * (torch.tensor([1.0, 1.0, -1.0]).to(self.device))[None, None, :]
           this_input_xyz = this_input_xyz[this_mask].unsqueeze(0)
           
           this_loss_chamfer = self.loss_chamfer(this_render_xyz, this_input_xyz)
           if not torch.isnan(this_loss_chamfer):
               self.loss += this_loss_chamfer
       
       
       print(self.loss)  
       return res
   
   def loss_backward(self, optimizer_list):
       if not isinstance(self.loss, float):
           self.loss.backward()
           for optimizer in optimizer_list:
               optimizer.step()
       #print("joints_pos_grad", self.joints_pos.grad)
       
       #print("joint1_grad", self.joint1.grad)
#       print("joint2_grad", self.joint2.grad)
#       print("joint3_grad", self.joint3.grad)
#       print("joint4_grad", self.joint4.grad)
#       print("joint5_grad", self.joint5.grad)
#       print("joint6_grad", self.joint6.grad)
#       print("joint7_grad", self.joint7.grad)
   
   def set_camera_intrinsics(self, K, width, height):
       self.depth_width = width
       self.depth_height = height
       self.proj_T = projectiveprojection_real(cam=K, x0=0.0, y0=0.0, w=self.depth_width, h=self.depth_height, nc=0.01, fc=10.0).unsqueeze(0)
       
   # Set optimizer
   def set_optimizer(self, quaternion, translation, joints_pos):
       self.quaternion = quaternion
       self.translation = translation
       self.joints_pos = joints_pos
       self.RT_optimizer = torch.optim.Adam(params=[self.quaternion, self.translation], lr=self.RT_lr)
       self.GA_optimizer = torch.optim.Adam(params=[self.joints_pos], lr=self.GA_lr)

   def set_RT_optimizer(self, quaternion, translation):
       self.quaternion = quaternion
       self.translation = translation
       
       self.RT_optimizer = torch.optim.Adam(params=[self.quaternion, self.translation], lr=self.RT_lr)
   
   def set_all_optimizer(self, joints_angle_pred, quaternion, translation, dr_order="single"):
       # quaternion : B x 4
       # translation : B x 3
       self.quaternion = quaternion
       self.translation = translation
       
       self.quaternion.requires_grad = True
       self.translation.requires_grad = True
       self.RT_optimizer = torch.optim.Adam(params=[self.quaternion, self.translation], lr=self.RT_lr)
       
       # joints_angle_pred : B x 7 x 1
       self.joint1 = joints_angle_pred[:, 0:1, :]
       self.joint2 = joints_angle_pred[:, 1:2, :]
       self.joint3 = joints_angle_pred[:, 2:3, :]
       self.joint4 = joints_angle_pred[:, 3:4, :]
       self.joint5 = joints_angle_pred[:, 4:5, :]
       self.joint6 = joints_angle_pred[:, 5:6, :]
       self.joint7 = joints_angle_pred[:, 6:7, :]
       
       self.joint1.requires_grad = True
       self.joint2.requires_grad = True
       self.joint3.requires_grad = True
       self.joint4.requires_grad = True
       self.joint5.requires_grad = True
       self.joint6.requires_grad = True
       self.joint7.requires_grad = True
       
       self.GA_joint1_optimizer = torch.optim.Adam([{'params': self.joint1, 'lr': self.GA_lr}])
       self.GA_joint2_optimizer = torch.optim.Adam([{'params': self.joint2, 'lr': self.GA_lr}])
       self.GA_joint3_optimizer = torch.optim.Adam([{'params': self.joint3, 'lr': self.GA_lr}])
       self.GA_joint4_optimizer = torch.optim.Adam([{'params': self.joint4, 'lr': self.GA_lr}])
       self.GA_joint5_optimizer = torch.optim.Adam([{'params': self.joint5, 'lr': self.GA_lr}])
       self.GA_joint6_optimizer = torch.optim.Adam([{'params': self.joint6, 'lr': self.GA_lr}])
       self.GA_joint7_optimizer = torch.optim.Adam([{'params': self.joint7, 'lr': self.GA_lr}])
       
       if dr_order == "single":
           self.GA_joint_dict = {0 : [self.RT_optimizer], # any one is ok
                             1 : [self.GA_joint1_optimizer],
                             2 : [self.GA_joint2_optimizer],
                             3 : [self.GA_joint3_optimizer],
                             4 : [self.GA_joint4_optimizer], 
                             5 : [self.GA_joint5_optimizer],
                             6 : [self.GA_joint6_optimizer],
                             7 : [self.GA_joint7_optimizer],
                             8 : [self.GA_joint7_optimizer] ,
                             "whole" : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer, self.GA_joint3_optimizer, 
                                       self.GA_joint4_optimizer, self.GA_joint5_optimizer, self.GA_joint6_optimizer, self.GA_joint7_optimizer
                                       ]
                            }
       elif dr_order == "sequence":
           self.GA_joint_dict = {0 : [self.RT_optimizer], # any one is ok
                                 1 : [self.RT_optimizer, self.GA_joint1_optimizer],
                                 2 : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer],
                                 3 : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer, self.GA_joint3_optimizer],
                                 4 : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer, self.GA_joint3_optimizer, self.GA_joint4_optimizer], 
                                 5 : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer, self.GA_joint3_optimizer, self.GA_joint4_optimizer, self.GA_joint5_optimizer],
                                 6 : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer, self.GA_joint3_optimizer, self.GA_joint4_optimizer, self.GA_joint5_optimizer, self.GA_joint6_optimizer],
                                 7 : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer, self.GA_joint3_optimizer, self.GA_joint4_optimizer, self.GA_joint5_optimizer, self.GA_joint6_optimizer, self.GA_joint7_optimizer],
                                 8 : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer, self.GA_joint3_optimizer, self.GA_joint4_optimizer, self.GA_joint5_optimizer, self.GA_joint6_optimizer, self.GA_joint7_optimizer] ,
                                 "whole" : [self.RT_optimizer, self.GA_joint1_optimizer, self.GA_joint2_optimizer, self.GA_joint3_optimizer, 
                                       self.GA_joint4_optimizer, self.GA_joint5_optimizer, self.GA_joint6_optimizer, self.GA_joint7_optimizer
                                       ]
                                }
       else:
           raise ValueError
       
   

   def RT_optimizer_zero_grad(self):
       self.RT_optimizer.zero_grad()
   
   def RT_optimizer_step(self):
       self.RT_optimizer.step()

   
   def GA_optimizer_zero_grad(self):
       self.GA_optimizer.zero_grad()
   
   def GA_optimizer_step(self):
       self.GA_optimizer.step()
   
   def set_o6dpose_optimizer(self, o6dposes, translation, joints_pos):
       assert o6dposes.shape[-1] == 6
       self.o6dposes = o6dposes
       self.translation = translation
       self.joints_pos = joints_pos
       self.RT_optimizer = torch.optim.Adam(params=[self.o6dposes, self.translation], lr=self.RT_lr)
       self.GA_optimizer = torch.optim.Adam(params=[self.joints_pos], lr=self.GA_lr)
   
   def load_joints(self, joints_pos):
       self.joints_pos = joints_pos # B x N x 1
   
   def concat_and_sample_mesh(self, num_pts):
       self.vertices_list_, self.joints_x3d_rob, self.R2C_list, self.T_list = concat_part_mesh(self.vertices_list, self.joints_pos, self.device,True)
       self.vertices = torch.cat(self.vertices_list_,dim=1)
       _, N, _ = self.vertices.shape
       index = random.sample(range(N), num_pts)
       return self.vertices[:, index, :]
       
   def get_sample_index(self, num_pts):
       _, N, _ = self.single_ori_vertices.shape 
       index = random.sample(range(N), num_pts)
       return index
       
   def get_sample_meshes(self, joints_pos, index):
       self.joints_pos = joints_pos
       self.vertices_list_, self.joints_x3d_rob, self.R2C_list, self.T_list = concat_part_mesh(self.vertices_list, self.joints_pos, self.device,True)
       self.vertices = torch.cat(self.vertices_list_,dim=1)
       return self.vertices[:, index, :]

if __name__ == "__main__":
   cfg = dict()
   cfg["DR"] = dict()
   cfg["DR"].update({"CAD_MODEL_PATHS" : 
           [f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/Link0.obj", 
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/Link1.obj", 
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/Link2.obj",\
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/Link3.obj", 
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/Link4.obj", 
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/Link5.obj",\
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/Link6.obj", 
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/Link7.obj",  
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/panda_hand.obj",\
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/panda_finger_joint1.obj", 
            f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/DifferentiableRenderer/Kaolin/franka_panda/panda_finger_joint2.obj"
           ]})
   cfg["DR"].update({"RT_LR" : 0.005})
   cfg["DR"].update({"GA_LR" : 0.005})
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   DF = DiffPFDepthRenderer(cfg, device)
   DF.load_mesh()
   
   root_dir = "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/9_kinect_left_0/"
   
   camera_K = load_camera_intrinsics(os.path.join(root_dir, "_camera_settings.json"))
   K = torch.from_numpy(camera_K).float().to(device)
   print("camera_K",camera_K)
   height = 360
   width = 640
   DF.set_camera_intrinsics(K, width=width, height=height)
   
#   json_paths_list = [f"/mnt/hyperplane/yangtian/synthetic/Panda/panda_synth_train_dr/20995{i}.json" for i in range(1, 10, 2)]
#   for json_path in json_paths_list:
#       quaternion, translation, this_joint_angles = load_dream_data(json_path)
#       
#       this_joint_angles = torch.from_numpy(np.array(this_joint_angles)).float().to(device).reshape(1, 7, 1)
#       this_quaternion = torch.from_numpy(np.array(quaternion)).float().to(device).reshape(1, 4)
#       R_NORMAL_UE = np.array([
#                [0, -1, 0],
#                [0, 0, -1],
#                [1, 0, 0],
#            ])
#       R_NORMAL_UE = torch.from_numpy(R_NORMAL_UE).float().to(device)
#       this_rot = quaternion_to_matrix(this_quaternion) 
#       this_rot = this_rot @ R_NORMAL_UE
#       this_quaternion = matrix_to_quaternion(this_rot)
#       #this_quaternion = torch.from_numpy(this_quaternion.detach().cpu().numpy()).float().to(device)
#       this_translation = torch.from_numpy(np.array(translation)).float().to(device).reshape(1, 3)
#       
#       this_quaternion.requires_grad = True
#       this_translation.requires_grad = True     
#       
#       DF.set_all_optimizer(this_joint_angles, this_quaternion, this_translation)
#       DF.batch_mesh(1)
#   
#       DF.concat_mesh()
#       img = DF.Rasterize()[1][0]
#       print(torch.max(img))
#       img = img.detach().cpu().numpy() * (255)
#       
#       
#       json_idx = json_path.split('/')[-1][:6]
#       print(json_idx)
#       image_path = f"./data_imgs/{json_idx}.png"
#       cv2.imwrite(image_path, img)       
   pkl_path_list = glob.glob(os.path.join(root_dir, "*json"))
   pkl_path_list.sort()

   for pkl_path in tqdm(pkl_path_list[:-2]):
       pkl_idx = pkl_path.split('/')[-1][:6]
       image_path = os.path.join("/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/9_kinect_left_0/", f"{pkl_idx}_mask.exr")
       if os.path.exists(image_path):
           continue

       with open(pkl_path , "rb") as fh:
           #json_data = pickle.load(fh)[0]
           json_data = json.load(fh)[0]
           
           
       json_keypoints_data = json_data["keypoints"]                    
       json_joints_data = json_data["joints_3n_fixed_42"]
       json_joints_8_data = json_data["joints"] 
       
       this_joint_angles = [kp["position"] for idx, kp in enumerate(json_joints_8_data)]
       this_joint_angles = torch.from_numpy(np.array(this_joint_angles))[:7].float().to(device).reshape(1, 7, 1)
       this_rot = torch.from_numpy(np.array(json_keypoints_data[0]["R2C_mat"])).float().to(device).reshape(1, 3, 3)
       this_quaternion = matrix_to_quaternion(this_rot) # 1 x 4
       this_translation = torch.from_numpy(np.array(json_keypoints_data[0]["location_wrt_cam"])).float().to(device).reshape(1, 3) # 1 x 3
       kps_7_list = (np.array([kp["location_wrt_cam"] for idx, kp in enumerate(json_keypoints_data)])[[0,2,3,4,6,7,8]])
       kps_14_list = np.array([json_joints_data[idx]["location_wrt_cam"] for idx in range(len(json_joints_data))])
       
       
       kps_7_list = (camera_K @ kps_7_list.T).T # (N, 3)
       kps_7_list = (kps_7_list / kps_7_list[:, -1:])[:, :2]
       kps_14_list = (camera_K @ kps_14_list.T).T # (N, 3)
       kps_14_list = (kps_14_list / kps_14_list[:, -1:])[:, :2]
       
       this_quaternion.requires_grad = True
       this_translation.requires_grad = True
#       this_joint_angles.requires_grad = True
       
       DF.set_all_optimizer(this_joint_angles, this_quaternion, this_translation)
       DF.batch_mesh(1)
   
       DF.concat_mesh()
       #img = DF.Rasterize()[0][0].detach().cpu().numpy() * (-255)
       img = DF.Rasterize()[1][0].detach().cpu().numpy()
       print(img.shape)
       print(image_path)
       #image_path = f"./data_imgs/render_depth_{str(b).zfill(4)}.png"
       cv2.imwrite(image_path, img)
       
#       images = []
#       for n in range(len(kps_7_list)):
#           image = overlay_points_on_image(image_path, [kps_7_list[n]], annotation_color_dot = ["yellow"], point_diameter=4)
#           images.append(image)
#       
#       img = mosaic_images(
#                   images, rows=2, cols=4, inner_padding_px=10
#               )
#       save_path = image_path.replace("render", "7_kps_render.png")
#       img.save(save_path)
#       
#       images = []
#       for n in range(len(kps_14_list)):
#           image = overlay_points_on_image(image_path, [kps_14_list[n]], annotation_color_dot = ["blue"], point_diameter=4)
#           images.append(image)
#       
#       img = mosaic_images(
#                   images, rows=4, cols=4, inner_padding_px=10
#               )
#       save_path = image_path.replace("render", "14_kps_render.png")
#       img.save(save_path)       
##   
   
   
#   json_path = f"/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/depth_c2rp/info.json"
#   json_in = open(json_path, 'r')
#   json_data = json.load(json_in)
#   poses_list = json_data["pose"]
#   joint_angles_list = json_data["joints"]
#   kps_7_list = json_data["kps_7"]
#   kps_14_list = json_data["kps_14"]
#   
#   for b in range(len(poses_list)):
#       this_pose = poses_list[b]
#       this_joint_angles = joint_angles_list[b]
#       if b == 1:
#           print(kps_7_list[b])
#       
#       this_pose = torch.from_numpy(np.array(this_pose)).float().to(device).reshape(1, 4, 4)
#       this_joint_angles = torch.from_numpy(np.array(this_joint_angles)).float().to(device).reshape(1, 7, 1)
       
       

       
#       this_quaternion = matrix_to_quaternion(this_pose[:, :3, :3]) # 1 x 4
#       this_translation = this_pose[:, :3, 3] # 1 x 3
       
#       this_quaternion.requires_grad = True
#       this_translation.requires_grad = True
##       this_joint_angles.requires_grad = True
#       
#       DF.set_all_optimizer(this_joint_angles, this_quaternion, this_translation)
#       DF.batch_mesh(1)
#   
#       DF.concat_mesh()
#       img = DF.Rasterize()[0].detach().cpu().numpy() * (-255)
#       
#       image_path = f"./data_imgs/render_depth_{str(b).zfill(4)}.png"
#       cv2.imwrite(image_path, img)
#       images = []
#       for n in range(len(kps_7_list[b])):
#           image = overlay_points_on_image(image_path, [kps_7_list[b][n]], annotation_color_dot = ["yellow"], point_diameter=4)
#           images.append(image)
#       
#       img = mosaic_images(
#                   images, rows=2, cols=4, inner_padding_px=10
#               )
#       save_path = image_path.replace("render", "7_kps_render.png")
#       img.save(save_path)
#       
#       images = []
#       for n in range(len(kps_14_list[b])):
#           image = overlay_points_on_image(image_path, [kps_14_list[b][n]], annotation_color_dot = ["blue"], point_diameter=4)
#           images.append(image)
#       
#       img = mosaic_images(
#                   images, rows=4, cols=4, inner_padding_px=10
#               )
#       save_path = image_path.replace("render", "14_kps_render.png")
#       img.save(save_path)
       
   
#   quaternion = torch.tensor([ 0.6257,  0.7686,  0.0969, -0.0913],device=device).reshape(1, 4)
#   translation = torch.tensor([-0.37863880349038576, 0.2666280120354375, 1.5995485870204855], device=device).reshape(1, 3)
#   joints_angle = torch.tensor([-0.6526082062025893, 
#                                             0.9279837857801965, 
#                                             2.551399836921973, 
#                                             -2.33985123801545, \
#                                             1.4105617980583107, 
#                                             2.125588105143108, 
#                                             1.2248684962301084, 
#                                            ], device=device).reshape(1, 7, 1)
#   joints_angle.requires_grad = True
#   quaternion.requires_grad = True
#   translation.requires_grad = True
   


#   DF.set_optimizer(quaternion, translation, joints_angle)
#   DF.batch_mesh(1)
#   
#   DF.concat_mesh()
#   img = DF.Rasterize()
#   print(img.shape)
#   print(torch.max(img))
#   print(torch.min(img))
#   
#   cv2.imwrite(f"./render_depth.png", img[0].detach().cpu().numpy() * (-255))
#   
#   pcs = img[0].detach().cpu().numpy().reshape(-1, 3)
#   #pcs[:, 1] *= -1
#   pcs[:, 2] *= -1
#   np.savetxt(f"./render_pcs.txt", pcs)
#   
#   gt_depth = f"./0000_depth_60.exr"
#   gt_img = cv2.imread(gt_depth, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32) 
#   K = K.detach().cpu().numpy()
#   gt_pcs = depthmap2pointcloud(gt_img, K[0][0], K[1][1], K[0][2], K[1][2])
#   np.savetxt(f"./gt_pcs.txt", gt_pcs)
   
   
#   gt_rgb_image = cv2.imread(img_path[b])
#            gt_rgb_image = cv2.cvtColor(gt_rgb_image, cv2.COLOR_RGB2BGR)
#            gt_rgb_image = cv2.resize(gt_rgb_image, (depth_valid_render_mask.shape[2], self.render_depth_mask.shape[1]), interpolation=cv2.INTER_CUBIC)[12:-12, :, :]
#             
#   gt_depth_image = PILImage.fromarray((cv2.imread(gt_depth, cv2.IMREAD_UNCHANGED)* 255).astype('uint8') ).convert('RGB')
#   render_depth_image = PILImage.fromarray((img[0, : ,:, -1].detach().cpu().numpy() * (-255)).astype('uint8')).convert('RGB')
#   blend_image = PILImage.blend(render_depth_image, gt_depth_image, 0.5)
#   blend_image.save(f"./blend.png")
    
   
#    depth_gt = np.loadtxt(f"./gt/depth_robot_gt.txt")
#    depth_gt_tensor = torch.from_numpy(depth_gt.reshape(1, height,width, -1)).to(device) 
#    depth_gt_mask = np.loadtxt(f"./gt/mask_robot_gt.txt")
#    depth_gt_mask_tensor = torch.from_numpy(depth_gt_mask.reshape(1, height,width, -1)).float().to(device) 
#    depth_gt_mask_tensor.requires_grad = False

   
#    num = 10
#    for i in range(num):
#        DF.RT_optimizer_zero_grad()
#        DF.GA_optimizer_zero_grad()
#        
#        DF.concat_mesh()
#        DF.Rasterize()
#        DF.loss_forward(depth_gt_tensor, depth_gt_mask_tensor)
#        DF.loss_backward()
#        
#        DF.RT_optimizer_step()
#        DF.GA_optimizer_step()
   


















