import kaolin as kal
import torch
import torch.nn as nn
import math
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
from PIL import Image as PILImage
from Render_utils import projectiveprojection_real, euler_angles_to_matrix, load_part_mesh, concat_part_mesh, exists_or_mkdir
from Render_utils import quaternion_to_matrix, matrix_to_quaternion, euler_angles_to_matrix, matrix_to_euler_angles, seg_and_transform
from tqdm import tqdm
import argparse
import os

class DiffPFDepthRenderer():
    def __init__(self, cfg, device):
        super().__init__()
        """
        expect cfg is a dict
        """
        self.device = device
#        self.CAD_model_paths = cfg.DR.CAD_model_paths
#        self.RT_lr = cfg.DR.RT_lr
#        self.GA_lr = cfg.DR.GA_lr
        self.CAD_model_paths = cfg["DR"]["CAD_MODEL_PATHS"]
        self.RT_lr = cfg["DR"]["RT_LR"]
        self.GA_lr = cfg["DR"]["GA_LR"]
        self.loss_fn = nn.MSELoss()
        #self.basis_change = torch.tensor([[[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]]],device=self.device)
    
    def load_mesh(self):
        self.vertices_list, self.faces_list, self.normals_list, self.faces_num_list = load_part_mesh(self.CAD_model_paths, self.device)
        self.single_face_indexs = torch.from_numpy(np.array(self.faces_num_list)).reshape(1, -1, 3, 3).float().to(self.device).contiguous()
        self.single_basis_change = torch.tensor([[[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]]],device=self.device)
        #self.face_indexs = self.face_indexs.repeat(bs, 1, 1, 1) # B x num_faces x 3 x 3
        self.single_ori_vertices = torch.cat(self.vertices_list,dim=1)
        #print("self.ori_vertices", self.ori_vertices.shape)
    
    def batch_mesh(self, bs):
        self.ori_vertices = self.single_ori_vertices.repeat(bs, 1, 1)
        self.face_indexs = self.single_face_indexs.repeat(bs, 1, 1, 1)
        self.basis_change = self.single_basis_change.repeat(bs, 1, 1)
#        print("self.basis_change", self.basis_change.shape)
        #print("self.ori_vertices.shape", self.ori_vertices.shape)
        #print("self.face_indexs.shape", self.face_indexs.shape)        
    
    def concat_mesh(self):
        self.vertices_list_, self.joints_x3d_rob, self.R2C_list, self.T_list = concat_part_mesh(self.vertices_list, self.joints_pos, self.device,True)
        self.vertices = torch.cat(self.vertices_list_,dim=1)
        self.faces = torch.cat(self.faces_list,dim=0)
        
        self.quaternion_norm = self.quaternion / torch.norm(self.quaternion, dim=-1,keepdim=True)
        self.Rot_matrix = quaternion_to_matrix(self.quaternion_norm)
#        print("self.translation.shape", self.translation.shape)
#        print("self.joints_x3d_rob", self.joints_x3d_rob.shape)
#        print("self.Rot_matrix.shape", self.Rot_matrix.shape)
#        self.joints_x3d_cam = (torch.bmm(self.Rot_matrix, self.joints_x3d_rob.permute(0,2,1))).permute(0,2,1) + self.translation[:, None, :]
#        np.savetxt(f"./rendered_cam.txt", self.joints_x3d_cam.reshape(-1,3).detach().cpu().numpy())
        
#        print("self.Rot_matrix", self.Rot_matrix.shape)
#        print("self.vertices.shape", self.vertices.shape)
#        print("self.basis_change.shape", self.basis_change.shape)
#        print("self.translation.shape", self.translation.shape)
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
        
#        print("self.Rot_matrix", self.Rot_matrix.shape)
#        print("self.basis_change", self.basis_change.shape)
        
        self.render_depth_image = seg_and_transform(im_features, self.render_depth_mask, \
                                                 self.Rot_matrix,self.basis_change,self.translation, self.device,self.R2C_list,self.T_list)
        
#        for b in range(self.render_depth_image.shape[0]):
#            np.savetxt(f"./check_depths/rendered_{b}_xyz.txt", self.render_depth_image[b].reshape(-1,3).detach().cpu().numpy() * np.array([1.0,-1.0,-1.0]))
#            cv2.imwrite(f"./check_depths/rendered_{b}_depth.png", (self.render_depth_image[b][:, :, 2:3]*(-255)).detach().cpu().numpy())
    
    def loss_forward(self, input_depth, input_mask,img_path=None,update_idx=None):
        valid_mask = (input_mask > 1e-3)
        depth_valid_render_mask = (self.render_depth_mask > 1e-3) # B x H x W x 3
        test_mask = valid_mask * depth_valid_render_mask
        
#        print("valid_mask", valid_mask.device)
#        print("depth_valid_render_mask", depth_valid_render_mask.device)
#        print("self.render_depth_image", self.render_depth_image.device)
#        print("input_depth", input_depth.device)
        all_depth = self.render_depth_image * test_mask
        all_input = input_depth * test_mask
        
#        save_path = f"./check_depths/{update_idx}_imgs"
#        exists_or_mkdir(save_path)
#        print("self.render_depth_image.shape", self.render_depth_image.shape)
#        B, H, W, C = self.render_depth_image.shape
#        res = np.zeros_like(self.render_depth_image.detach().cpu().numpy())
#        
#        for b in range(self.render_depth_image.shape[0]):
##            np.savetxt(f"./check_depths/rendered_{b}_xyz.txt", self.render_depth_image[b].reshape(-1,3).detach().cpu().numpy() * np.array([1.0,-1.0,-1.0]))          
#            
#            render_mask_image = PILImage.fromarray((depth_valid_render_mask[b].detach().cpu().numpy()).astype(np.uint8) * 255)
#            gt_rgb_image = PILImage.open(img_path[b]).convert("RGB")
#            blend_image = PILImage.blend(render_mask_image, gt_rgb_image, 0.7)
#            #blend_image.save(os.path.join(save_path, f"{b}_blend.png"))
#            blend_image_np = np.array(blend_image)
#            res[b] = blend_image_np
            #print("blend_image_np.shape", blend_image_np.shape)
#            cv2.imwrite(f"./check_depths/rendered_{b}_masked_depth.png", (all_depth[b][:, :, 2:3]*(-255)).detach().cpu().numpy())
#            cv2.imwrite(f"./check_depths/rendered_{b}_depth.png", (self.render_depth_image[b][:, :, 2:3]*(-255)).detach().cpu().numpy())
#            cv2.imwrite(f"./check_depths/gt_{b}_masked_depth.png", all_input[b].detach().cpu().numpy() * 255)
#            cv2.imwrite(f"./check_depths/gt_{b}_depth.png", input_depth[b].detach().cpu().numpy() * 255)
#            cv2.imwrite(f"./check_depths/render_{b}_mask.png", depth_valid_render_mask[b].detach().cpu().numpy() * 255)
#            cv2.imwrite(f"./check_depths/predict_{b}_mask.png", valid_mask[b].detach().cpu().numpy() * 255)
            
        self.loss = self.loss_fn((self.render_depth_image * test_mask)[:, :, :, 2:3].float() * (-1.0), (input_depth * test_mask).float())
        print(self.loss)  
        return res
    
    def loss_backward(self):
        self.loss.backward()
    
    def set_camera_intrinsics(self, K, width, height):
        self.depth_width = width
        self.depth_height = height
        self.proj_T = projectiveprojection_real(cam=K, x0=0.0, y0=0.0, w=self.depth_width, h=self.depth_height, nc=0.01, fc=10.0).unsqueeze(0)
        
    # Set optimizer
    def set_optimizer(self, quaternion, translation, joints_pos):
        self.quaternion = quaternion
        self.translation = translation
        self.joints_pos = joints_pos
#        self.gripper_dis = self.joints_pos[:, 7:8, 0]
#        self.joints_angle = self.joints_pos[:, :7, 0]
#        print("self.gripper_dis",self.gripper_dis.shape)
#        print("self.joints_angle.shape", self.joints_angle.shape)
        self.RT_optimizer = torch.optim.Adam(params=[self.quaternion, self.translation], lr=self.RT_lr)
        self.GA_optimizer = torch.optim.Adam(params=[self.joints_pos], lr=self.GA_lr)
        
    def RT_optimizer_zero_grad(self):
        self.RT_optimizer.zero_grad()
    
    def RT_optimizer_step(self):
        self.RT_optimizer.step()
    
    def GA_optimizer_zero_grad(self):
        self.GA_optimizer.zero_grad()
    
    def GA_optimizer_step(self):
        self.GA_optimizer.step()
    
    def load_joints(self, joints_pos):
        self.joints_pos = joints_pos # B x N x 1
    
    def concat_and_sample_mesh(self, num_pts):
        self.vertices_list_, self.joints_x3d_rob, self.R2C_list, self.T_list = concat_part_mesh(self.vertices_list, self.joints_pos, self.device,True)
        self.vertices = torch.cat(self.vertices_list_,dim=1)
        print("self.vertices.shape", self.vertices.shape)
    

if __name__ == "__main__":
    cfg = dict()
    cfg.update({"DR": {"CAD_MODEL_PATHS" : [f"./franka_panda/Link0.obj", f"./franka_panda/Link1.obj", f"./franka_panda/Link2.obj",\
             f"./franka_panda/Link3.obj", f"./franka_panda/Link4.obj", f"./franka_panda/Link5.obj",\
             f"./franka_panda/Link6.obj", f"./franka_panda/Link7.obj", f"./franka_panda/panda_hand.obj",\
             f"./franka_panda/panda_finger_joint1.obj", f"./franka_panda/panda_finger_joint2.obj"
            ], "RT_LR" : 0.005, "GA_LR" : 0.005}})
#    cfg.update({"DR" : {"RT_lr" : 0.005}})
#    cfg.update({"DR" : {"GA_lr" : 0.005}})
    print(cfg)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    DF = DiffPFDepthRenderer(cfg, device)
    DF.load_mesh()
    DF.batch_mesh(1)
    K = torch.tensor([
                    [502.30, 0.0, 319.5],
                    [0.0, 502.30, 179.5],
                    [0.0, 0.0, 1.0]
                    ], device=device)
    height = 360
    width = 640
    #DF.set_camera_intrinsics(K, width=width, height=height)
    
    quaternion = torch.tensor([ 0.6419,  0.7245, -0.2030,  0.1476],device=device,requires_grad=True)
    translation = torch.tensor([-0.30605295346808126,-0.2864948368273095,-1.404889194692222], device=device, requires_grad=True)
    #noise =  2 * 0.1 * torch.rand(translation.size()).to(device) - 0.1
    #translation = translation + noise
    #translation.requires_grad = True
    
    
    
    #gripper_dis = torch.from_numpy(np.array([0.013713414900289229])).float().to(device)
    joints_angle = torch.from_numpy(np.array([1.7010560821567642, 0.27977565142150723, -2.761883936874502, -2.430359192417069, \
                                         1.1527790406734584, 2.61622873174606, 0.2907628764894636, 0.013713414900289229
                                        ])).float().to(device)
    
    #gripper_dis = gripper_dis.unsqueeze(0)
    joints_angle = joints_angle[None, :, None]
    #gripper_dis.requires_grad = True
    joints_angle.requires_grad = True
    
    DF.load_joints(joints_angle)
    DF.concat_and_sample_mesh(num_pts=1000)
#    DF.set_optimizer(quaternion, translation, gripper_dis, joints_angle)
    
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
    


















