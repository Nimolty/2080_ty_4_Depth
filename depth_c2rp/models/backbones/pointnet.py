import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_softmax, scatter_max, scatter_log_softmax
import time


class PointNet2Stage(nn.Module):
    def __init__(self, input_channels=6, output_channels=256, gf_dim=64):
        super(PointNet2Stage, self).__init__()
        self.input_channels = input_channels
        self.gf_dim = gf_dim

        self.point_lin1 = nn.Linear(self.input_channels, self.gf_dim, bias=True)
        self.point_lin2 = nn.Linear(self.gf_dim, output_channels//2, bias=True)
        self.vox_lin1 = nn.Linear(output_channels//2, output_channels//2, bias=True)
        
        self.point_lin3 = nn.Linear(output_channels, output_channels, bias=True)
        self.point_lin4 = nn.Linear(output_channels, output_channels, bias=True)
        self.vox_lin2 = nn.Linear(output_channels, output_channels, bias=True)


    def forward(self, inp_feat, vox2point_idx):
        # per point features
        t1 = time.time()
        point_feat1 = F.relu(self.point_lin1(inp_feat), inplace=True)
        point_feat2 = F.relu(self.point_lin2(point_feat1), inplace=True)
        # maxpool point feats inside each occupied voxel to get occ_voxel feat
        t2 = time.time()
        #print("point_feat2.shape", point_feat2.shape)
        #print("vox2point_idx.shape", vox2point_idx.shape)
        occ_voxel_feat = scatter(point_feat2, vox2point_idx, dim=0, reduce='max')
        #print("occ_voxel_feat.shape", occ_voxel_feat.shape)
        t3 = time.time()
        occ_voxel_feat = F.relu(self.vox_lin1(occ_voxel_feat), inplace=True)
        # append vox feat to point feat
        point_global_feat = occ_voxel_feat[vox2point_idx]
        point_feat3 = torch.cat((point_global_feat,point_feat2),-1)
        point_feat4 = F.relu(self.point_lin3(point_feat3))
        point_feat5 = F.relu(self.point_lin4(point_feat4))
        # maxpool point feats inside each occupied voxel to get occ_voxel feat
        t4 = time.time()
        occ_voxel_feat2 = scatter(point_feat5, vox2point_idx, dim=0, reduce='max')
        occ_voxel_feat2 = F.relu(self.vox_lin2(occ_voxel_feat2))
        t5 = time.time()
        
#        print("t5 - t4", t5 - t4)
#        print("t4 - t3", t4 - t3)
#        print("t3 - t2", t3 - t2)
#        print("t2 - t1", t2 - t1)

        return occ_voxel_feat2


class Bilinear(nn.Module):
    def __init__(self, linear_chans):
        super(Bilinear, self).__init__()
        self.linear_module1 = nn.Sequential(
            nn.Linear(linear_chans, linear_chans),
            # nn.BatchNorm1d(linear_chans),
            nn.ReLU(),
            nn.Dropout()
        )
        self.linear_module2 = nn.Sequential(
            nn.Linear(linear_chans, linear_chans),
            # nn.BatchNorm1d(linear_chans),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self,inp):
        hidden1 = self.linear_module1(inp)
        hidden2 = self.linear_module2(hidden1)
        out = hidden2 + inp
        return out

class PointNetRefine(nn.Module):
    def __init__(self, input_channels=6, output_channels=128):
        super(PointNetRefine, self).__init__()
        self.pred_pos_enc = PointNetSimple(input_channels=input_channels, output_channels=output_channels)
        self.bilin1 = Bilinear(linear_chans=output_channels*2)
        self.bilin2 = Bilinear(linear_chans=output_channels*2)
        self.final_lin = nn.Linear(output_channels*2, output_channels)

    def forward(self, pred_inp, intersect_voxel_feat_end):
        pred_feat = self.pred_pos_enc(pred_inp)
        out = torch.cat((pred_feat, intersect_voxel_feat_end),1)
        out = self.bilin1(out)
        out = self.bilin2(out)
        out = self.final_lin(out)
        return out

