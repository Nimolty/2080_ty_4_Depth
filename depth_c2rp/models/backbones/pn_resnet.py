from __future__ import print_function

import os
import sys
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as tviz_models
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sample_pts(heatmap_pred, input_xyz, num_pts, threshold=0.1, z_min=0.01, z_max=3.0):
    # attributes
    # heatmap_pred :B x C x H x W
    # input_xyz : B x H x W x 3
    B, C, H, W = heatmap_pred.shape
    device = heatmap_pred.device
    
    sampling_pts = torch.zeros(B, C, num_pts, 3).to(device)
    
    for b in range(B):
        this_mask_xyz = (input_xyz[b][:, :, 2] > z_min) * (input_xyz[b][:, :, 2] < z_max)
        for c in range(C):
            this_mask = (heatmap_pred[b][c] >= threshold) * this_mask_xyz
            this_mask_v, this_mask_u = torch.where(this_mask == True)
            
            #print(len(this_mask_v))
            
            if len(this_mask_v) == 0:
                this_mask_v, this_mask_u = np.random.choice(np.arange(H), num_pts, replace=True), np.random.choice(np.arange(W), num_pts, replace=True)
            elif len(this_mask_v) < num_pts:
                random_index = np.random.choice(np.arange(len(this_mask_v)), num_pts, replace=True)
                this_mask_v, this_mask_u = this_mask_v[random_index], this_mask_u[random_index]
            else:
                random_index = np.random.choice(np.arange(len(this_mask_v)), num_pts, replace=False)
                this_mask_v, this_mask_u = this_mask_v[random_index], this_mask_u[random_index]
            
            sampling_pts[b][c] = input_xyz[b][[this_mask_v, this_mask_u]]
    return sampling_pts

class ResnetSimple(nn.Module):
    def __init__(
        self, n_keypoints=7, freeze=False, pretrained=True, full=False, 
    ):
        super(ResnetSimple, self).__init__()
        net = tviz_models.resnet101(pretrained=pretrained)
        self.full = full        
        #self.conv1 = net.conv1
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # upconvolution and final layer
        BN_MOMENTUM = 0.1
        if not full:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2048,out_channels=256,kernel_size=4,stride=2,padding=1,output_padding=0,),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=4,stride=2,padding=1,output_padding=0,),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=4,stride=2,padding=1,output_padding=0,),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=4,stride=2,padding=1,output_padding=0,),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )
        

    def forward(self, x):
        # x.shape : B x 3 x H x W
        
        # get 2D heatmaps
        hm = self.conv1(x)
        hm = self.bn1(hm)
        hm = self.relu(hm)
        hm = self.maxpool(hm)

        hm = self.layer1(hm)
        hm = self.layer2(hm)
        hm = self.layer3(hm)
        hm = self.layer4(hm)

        hm = self.upsample(hm)
        return hm

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        # expect
        n_pts = x.size()[1]
        x = x.transpose(2, 1) # 
        x = F.relu(self.bn1(self.conv1(x)))

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x

class Global_PointNet3D(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(Global_PointNet3D, self).__init__()
        self.global_feat = PointNetfeat(global_feat=True) 
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape : B x n_kps x n_pts x 3
        B, n_kps, n_pts, _ = x.shape
        global_x = x.view(B, -1, 3).contiguous()
        global_feat = self.global_feat(global_x).unsqueeze(1).repeat(1, n_kps, 1) # B x n_kps x 1024 : global features 
        
        out = global_feat.view(B * n_kps, -1) # B x n_kps x 2048
        #print(out.shape)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.dropout(self.fc2(out))))
        out = F.relu(self.bn3(self.dropout(self.fc3(out))))
        out = self.fc4(out)
        return out.view(B, n_kps, -1)
       
class PointNet3D(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNet3D, self).__init__()
        self.global_feat = PointNetfeat(global_feat=True) 
        self.local_feat = PointNetfeat(global_feat=True) 
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape : B x n_kps x n_pts x 3
        B, n_kps, n_pts, _ = x.shape
        local_feat = self.local_feat(x.view(-1, n_pts, 3).contiguous()).view(B, n_kps, -1) # B x n_kps x 1024 : local features for each keypoints 
        global_samples = torch.from_numpy(np.random.choice(np.arange(n_pts), (B * n_kps * (n_pts // n_kps)), replace=True).reshape(B, n_kps, -1, 1)).repeat(1, 1, 1, 3).to(x.device)
        global_x = torch.gather(x, 2, global_samples).view(B, -1, 3).contiguous()
        global_feat = self.global_feat(global_x).unsqueeze(1).repeat(1, n_kps, 1) # B x n_kps x 1024 : global features 
        
        out = torch.cat([local_feat, global_feat], dim=-1).view(B * n_kps, -1) # B x n_kps x 2048
        #print(out.shape)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.dropout(self.fc2(out))))
        out = F.relu(self.bn3(self.dropout(self.fc3(out))))
        out = self.fc4(out)
        return out.view(B, n_kps, -1)

class DepthNet(nn.Module):
    def __init__(self, n_keypoints=7, freeze=False, pretrained=True, full=False, threshold=0.1, num_pts=1024, out_dim=3):
        super(DepthNet, self).__init__()
        
        self.hm_backbone = ResnetSimple(n_keypoints=n_keypoints, freeze=False, pretrained=True, full=False)
        self.pointnet_3d = PointNet3D(k=out_dim)
        self.num_pts = num_pts
        self.threshold = threshold
    
    def forward(self, input_xyz_ray_d):
        # input_xyz_ray_d : B x 6 x H x W
        B, _, H, W = input_xyz_ray_d.shape
        
        # get 2D Heatmaps:
        heatmap_pred = self.hm_backbone(input_xyz_ray_d)
        heatmap_pred = (F.interpolate(heatmap_pred, (H, W), mode='bicubic',
                                                      align_corners=False) + 1) / 2. # B x C x H x W
                                                      
        # get sampling pts
        sampling_peaks_pts = get_sample_pts(heatmap_pred, input_xyz_ray_d[:, :3, :, :].permute(0, 2, 3, 1), self.num_pts, threshold=self.threshold) # B x C x num_pts x 3
        sampling_peaks_pts_mean = torch.mean(sampling_peaks_pts, dim=2, keepdims=True) # B x C x 1 x 3
        
        
        # get 3D Keypoints Predictions
        kps_3d_pred = self.pointnet_3d(sampling_peaks_pts - sampling_peaks_pts_mean) # B x C x 3
        kps_3d_pred = kps_3d_pred + sampling_peaks_pts_mean.squeeze(2)

               
        return heatmap_pred, kps_3d_pred

class PointResnet(nn.Module):
    def __init__(self, out_ch=3, global_flag=False):
        super(PointResnet, self).__init__()
        if global_flag:
            self.pointnet_3d = Global_PointNet3D(k=out_ch)
        else:
            self.pointnet_3d = PointNet3D(k=out_ch)
        
    def forward(self, sampling_pts):
        # sampling_pts.shape : B x n_kps x n_pts x 3
        sampling_pts_mean = torch.mean(sampling_pts, dim=2, keepdims=True)
        out_feats = self.pointnet_3d(sampling_pts - sampling_pts_mean) # B x C x out_dim
        
        return out_feats
           
        

if __name__ == "__main__":
    depthnet = DepthNet(n_keypoints=14, full=False, threshold=0.3,num_pts=1036,out_dim=3).cuda()
    input_xyz = torch.randn(1, 6, 384, 384).cuda()
    
    for i in range(1):
        t1 = time.time()
        depthnet(input_xyz)
        print(time.time() - t1)
    

