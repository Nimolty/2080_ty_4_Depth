import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import os
import sys
import time
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
#print(base_dir)

import numpy as np
import depth_c2rp.models.backbones.resnet_dilated as resnet_dilated
import depth_c2rp.models.backbones.pointnet as pnet
import depth_c2rp.models.backbones.pn_resnet as PointResnet
#from depth_c2rp.models.backbones.dream_hourglass import ResnetSimpleWoff

from depth_c2rp.models.heads import *
from depth_c2rp.models.backbones import *
from depth_c2rp.models.layers import *
from depth_c2rp.utils.spdh_network_utils import MLP_TOY
from depth_c2rp.utils.spdh_utils import compute_3n_loss_42_rob
from depth_c2rp.utils.spdh_sac_utils import compute_rede_rt
from depth_c2rp.voxel_utils.voxel_batch_utils import prepare_data, get_valid_points, get_occ_vox_bound, get_miss_ray, compute_ray_aabb, compute_gt,get_pred, compute_loss, get_embedding, adapt_lr
from depth_c2rp.voxel_utils.refine_batch_utils import get_pred_refine, compute_refine_loss

import Pointnet2_master.pointnet2.pointnet2_utils as ptn_utils
from Pointnet2_master.tools.pointnet2_msg import Pointnet2ClsMSG

head_names = {"FaPN": FaPNHead}
backbone_names = {"ResNet" : ResNet, "ResT" : ResT, "ConvNeXt" : ConvNeXt, "PoolFormer" : PoolFormer, 
                  "stacked_hourglass" : HourglassNet, "hrnet" : HRNet, "dreamhourglass_resnet_h" : ResnetSimple, "dreamhourglass_vgg" : DreamHourglass,
                  "dreamhourglass_resnet_woff_h": ResnetSimpleWoff, 
                 }
simplenet_names = {"Simple_Net" : MLP_TOY}


class build_voxel_simple_network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg["DATASET"]
        self.simple_net_params = {"dim" : self.dataset_cfg["NUM_JOINTS"] * 3, "h1_dim" :  1024, "out_dim" : 7}
        self.simplenet = simplenet_names[self.cfg["TOY_NETWORK"]](**self.simple_net_params)
    
    def forward(self, joints_3d_pred, joints_1d_gt=None, gt_angle_flag=False):
        # joints_3d_pred.shape : B x N x 3
        B, n_kps, _ = joints_3d_pred.shape
        joints_3d_pred_norm = joints_3d_pred - joints_3d_pred[:, :1, :].clone() # Normalization
        joints_angle_pred = self.simplenet(torch.flatten(joints_3d_pred_norm, 1)) # B x 7  
        # predict R and T using pose fitting
        all_dof_pred = joints_angle_pred[:, :, None] # B x 7 x 1
        
        if joints_1d_gt is not None and gt_angle_flag: 
            all_dof_pred = joints_1d_gt   
        joints_3d_rob_pred = compute_3n_loss_42_rob(all_dof_pred, joints_3d_pred.device)
        
        pose_pred_clone = []
        for b in range(B):
            pose_pred_clone.append(compute_rede_rt(joints_3d_rob_pred[b:b+1, :, :], joints_3d_pred_norm[b:b+1, :, :]))
        pose_pred_clone = torch.cat(pose_pred_clone)
        pose_pred_clone[:, :3, 3] = joints_3d_pred[:, 0] + pose_pred_clone[:, :3, 3] 
        
        return all_dof_pred, pose_pred_clone
        

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class IMNet(nn.Module):
    def __init__(self, inp_dim, out_dim, gf_dim=64, use_sigmoid=False):
        super(IMNet, self).__init__()
        self.inp_dim = inp_dim
        self.gf_dim = gf_dim
        self.use_sigmoid = use_sigmoid
        self.linear_1 = nn.Linear(self.inp_dim, self.gf_dim*4, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim*1, out_dim, bias=True)
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.normal_(self.linear_4.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_4.bias,0)

    def forward(self, inp_feat):
        l1 = self.linear_1(inp_feat)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        
        if self.use_sigmoid:
            l4 = self.sigmoid(l4)
        else:
            l4 = torch.max(torch.min(l4, l4*0.01+0.99), l4*0.01)
        
        return l4

class IEF(nn.Module):
    def __init__(self, device, inp_dim, out_dim, gf_dim=64, n_iter=3, use_sigmoid=False):
        super(IEF, self).__init__()
        self.device = device
        self.init_offset = torch.Tensor([0.001]).float().to(self.device)
        self.inp_dim = inp_dim
        self.gf_dim = gf_dim
        self.n_iter = n_iter
        self.use_sigmoid = use_sigmoid
        self.offset_enc = nn.Linear(1, 16, bias=True)
        self.linear_1 = nn.Linear(self.inp_dim+16, self.gf_dim*4, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim*1, out_dim, bias=True)
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.offset_enc.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.offset_enc.bias,0)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.normal_(self.linear_4.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_4.bias,0)


    def forward(self, inp_feat):
        batch_size = inp_feat.shape[0]
        # iterative update
        pred_offset = self.init_offset.expand(batch_size, -1)
        for i in range(self.n_iter):
            offset_feat = self.offset_enc(pred_offset)
            xc = torch.cat([inp_feat,offset_feat],1)
            l1 = self.linear_1(xc)
            l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

            l2 = self.linear_2(l1)
            l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

            l3 = self.linear_3(l2)
            l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

            l4 = self.linear_4(l3)
            pred_offset = pred_offset + l4
        
        if self.use_sigmoid:
            pred_offset = self.sigmoid(pred_offset)
        else:
            pred_offset = torch.max(torch.min(pred_offset, pred_offset*0.01+0.99), pred_offset*0.01)
        return pred_offset

def build_model(device,multires=8, multires_views=4, 
                rgb_in=3, rgb_out=32, 
                pnet_in=6, pnet_out=128, pnet_gf=32, roi_out_bbox=2,imnet_gf=64,n_iter=2,use_sigmoid=True,prob_out_dim=1
               ):
    # embedding network
    embed_fn, embed_ch = get_embedder(multires)
    embeddirs_fn, embeddirs_ch = get_embedder(multires_views) 
    
    # rgb network
    resnet_model=resnet_dilated.Resnet34_8s(inp_ch=rgb_in, out_ch=rgb_out) 
    
    # voxel-based pointnet network
    pnet_model = pnet.PointNet2Stage(input_channels=pnet_in,
                                    output_channels=pnet_out, gf_dim=pnet_gf)
    
    dec_inp_dim = pnet_out + rgb_out * (roi_out_bbox**2 + 36) \
                            + 2 * embed_ch + embeddirs_ch
    
    offset_dec = IEF(device, inp_dim=dec_inp_dim, out_dim=1, gf_dim=imnet_gf, 
                                    n_iter=n_iter, use_sigmoid=use_sigmoid).to(device)
    prob_dec = IMNet(inp_dim=dec_inp_dim, out_dim=prob_out_dim, 
                                gf_dim=imnet_gf, use_sigmoid=use_sigmoid).to(device)
    
    return embed_fn, embeddirs_fn, resnet_model, pnet_model, offset_dec,prob_dec 

            
class build_voxel_network(nn.Module):
    def __init__(self, cfg, device=torch.device("cpu")):
        super().__init__()
        
        # voxel network cfg
        self.cfg = cfg
        self.voxel_cfg = cfg["voxel_network"]
        if not self.voxel_cfg["raw_input_type"]:
            self.embed_ch_num = 2
        else:
            self.embed_ch_num = 3
        
        
        self.device = device
        
        # embedding network
        self.embed_fn, self.embed_ch = get_embedder(self.voxel_cfg["multires"])
        self.embeddirs_fn, self.embeddirs_ch = get_embedder(self.voxel_cfg["multires_views"]) 
        
        # rgb network (here is xyz network)
        if "Resnet34_8s" in self.voxel_cfg["rgb_model_type"]:
            self.resnet_model = resnet_dilated.Resnet34_8s(inp_ch=self.voxel_cfg["rgb_in"], out_ch=self.voxel_cfg["rgb_out"], global_ratio=self.voxel_cfg["global_ratio"], mid_channels=self.voxel_cfg["rgb_out"]*2, camera_intrin_aware=self.voxel_cfg["camera_intrin_aware"]) 
            
            if self.voxel_cfg["rgb_model_type"] == "Point_Resnet34_8s":
                self.point_resnet_model = PointResnet(out_ch=self.voxel_cfg["rgb_out"], global_flag=self.voxel_cfg["rgb_model_global_flag"])
        elif self.voxel_cfg["rgb_model_type"] == "PointResnet":
            self.resnet_model = PointResnet(out_ch=self.voxel_cfg["rgb_out"], global_flag=self.voxel_cfg["rgb_model_global_flag"])
        elif self.voxel_cfg["rgb_model_type"] == "Pointnet_pp":
            self.resnet_model = Pointnet2ClsMSG(0)
            self.pointnet2cls_ch = 1024
            
        
        if cfg["voxel_network"]["local_embedding_type"] == "ROIConcat":
            self.dec_inp_dim = self.voxel_cfg["pnet_out"] + self.voxel_cfg["rgb_out"] * (self.voxel_cfg["roi_out_bbox"]**2 + (cfg["DATASET"]["INPUT_RESOLUTION"][0] // self.voxel_cfg["global_ratio"]) **2) \
                            + self.embed_ch_num * self.embed_ch + self.embeddirs_ch
        elif cfg["voxel_network"]["local_embedding_type"] == "ROIPooling":
            self.dec_inp_dim = self.voxel_cfg["pnet_out"] + self.voxel_cfg["rgb_out"] * (1 + (cfg["DATASET"]["INPUT_RESOLUTION"][0] // self.voxel_cfg["global_ratio"]) **2) \
                            + self.embed_ch_num * self.embed_ch + self.embeddirs_ch
        elif cfg["voxel_network"]["local_embedding_type"] == "NOROI":
            self.dec_inp_dim = self.voxel_cfg["pnet_out"] + self.voxel_cfg["rgb_out"] * ((cfg["DATASET"]["INPUT_RESOLUTION"][0] // self.voxel_cfg["global_ratio"]) **2) \
                            + self.embed_ch_num * self.embed_ch + self.embeddirs_ch
        
        if self.voxel_cfg["rgb_model_type"] == "PointResnet":
            self.dec_inp_dim = self.voxel_cfg["pnet_out"] + self.voxel_cfg["rgb_out"] \
                            + self.embed_ch_num * self.embed_ch + self.embeddirs_ch
        if self.voxel_cfg["rgb_model_type"] == "Pointnet_pp":
            self.dec_inp_dim = self.voxel_cfg["pnet_out"] + self.pointnet2cls_ch \
                            + self.embed_ch_num * self.embed_ch + self.embeddirs_ch
        
        # voxel-based pointnet network
        self.pnet_model = pnet.PointNet2Stage(input_channels=self.voxel_cfg["pnet_in"],
                                    output_channels=self.voxel_cfg["pnet_out"], gf_dim=self.voxel_cfg["pnet_gf"])
        self.pnet_model = self.pnet_model.to(self.device)
        
        #self.dec_inp_dim = self.voxel_cfg["pnet_out"] + self.rgb_local_global_ch \
        #                    + self.embed_ch_num * self.embed_ch + self.embeddirs_ch
        
        self.offset_dec = IEF(self.device, inp_dim=self.dec_inp_dim, out_dim=1, gf_dim=self.voxel_cfg["imnet_gf"], 
                                    n_iter=self.voxel_cfg["n_iter"], use_sigmoid=self.voxel_cfg["use_sigmoid"]).to(self.device)
        self.prob_dec = IMNet(inp_dim=self.dec_inp_dim, out_dim=self.voxel_cfg["prob_out_dim"], 
                                gf_dim=self.voxel_cfg["imnet_gf"], use_sigmoid=self.voxel_cfg["use_sigmoid"]).to(self.device)
    
    def forward(self, batch, mode, epoch):
        # prepare data
        data_dict = prepare_data(batch)
        
        # get valid pts other than kps
        get_valid_points(data_dict, valid_sample_num=self.cfg["voxel_network"]["valid_sample_num"], rgb_in=self.cfg["voxel_network"]["rgb_in"])
        
        # get occupied voxel data
        get_occ_vox_bound(data_dict, res=self.cfg["voxel_network"]["res"])
        
        # get kps ray data
        get_miss_ray(data_dict)
        
        # ray AABB slab test
        intersect_pair_flag = compute_ray_aabb(data_dict) 
        
        # compute gt
        compute_gt(data_dict)
        
        # get embedding
        #get_embedding_ours(data_dict, embed_fn, embeddirs_fn, full_xyz_feat, pnet_model) 
        if self.voxel_cfg["rgb_model_type"] != "Point_Resnet34_8s":
            self.point_resnet_model = None
        if self.voxel_cfg["rgb_model_type"] == "PointResnet":
            self.point_resnet_model = self.resnet_model
            
        self.fps_npts = self.voxel_cfg["fps_npts"] if self.voxel_cfg["rgb_model_type"] == "Pointnet_pp" else None
        get_embedding(data_dict, self.embed_fn, self.embeddirs_fn, self.resnet_model, self.pnet_model, self.point_resnet_model, 
        rgb_global_flag=self.voxel_cfg["rgb_model_global_flag"], local_embedding_type=self.cfg["voxel_network"]["local_embedding_type"], roi_inp_bbox=self.cfg["voxel_network"]["roi_inp_bbox"], pnet_in_rgb_flag=self.cfg["voxel_network"]["pnet_in_rgb_flag"],
        roi_out_bbox=self.cfg["voxel_network"]["roi_out_bbox"], resnet_model_type=self.voxel_cfg["rgb_model_type"], fps_npts=self.fps_npts) 
        
        # get pred
        get_pred(data_dict, mode, epoch, self.offset_dec, self.prob_dec, self.device, raw_input_type=self.cfg["voxel_network"]["raw_input_type"])
        
        loss_dict = compute_loss(data_dict, mode, epoch,self.device) 
        
        return loss_dict, data_dict














class build_voxel_refine_network(nn.Module):
    def __init__(self, cfg, device=torch.device("cpu")):
        super().__init__()
        
        # voxel network cfg
        self.cfg = cfg
        self.voxel_cfg = cfg["voxel_network"]
        self.refine_voxel_cfg = cfg["refine_voxel_network"]
        self.device = device
        
        # positional embedding
        if self.refine_voxel_cfg["pos_encode"]:
            self.embed_fn, self.embed_ch = get_embedder(self.refine_voxel_cfg["multires"])
            self.embeddirs_fn, self.embeddirs_ch =get_embedder(self.refine_voxel_cfg["multires_views"])
        else:
            self.embed_fn, self.embed_ch = get_embedder(self.refine_voxel_cfg["multires"], i=-1)
            self.embeddirs_fn, self.embeddirs_ch = get_embedder(self.refine_voxel_cfg["multires_views"], i=-1)
            assert self.embed_ch == self.embeddirs_ch == 3
        
        # pointnet
        if self.refine_voxel_cfg["refine_pnet_model_type"] == 'twostage':
            self.pnet_model = pnet.PointNet2Stage(input_channels=self.refine_voxel_cfg["refine_pnet_in"],
                                output_channels=self.refine_voxel_cfg["refine_pnet_out"], gf_dim=self.refine_voxel_cfg["refine_pnet_gf"]).to(self.device)
        else:
            raise NotImplementedError('Does not support Pnet type for RefineNet: {}'.format(self.refine_voxel_cfg["refine_pnet_model_type"]))
        
        if not self.voxel_cfg["raw_input_type"]:
            self.embed_ch_num = 1
        else:
            self.embed_ch_num = 2
        
        # decoder input dim
        dec_inp_dim = self.refine_voxel_cfg["refine_pnet_out"] + self.embed_ch_num * self.embed_ch + self.embeddirs_ch
        
        if cfg["voxel_network"]["local_embedding_type"] == "ROIConcat":
            dec_inp_dim += self.voxel_cfg["rgb_out"] * (self.voxel_cfg["roi_out_bbox"]**2 + (cfg["DATASET"]["INPUT_RESOLUTION"][0] // self.voxel_cfg["global_ratio"]) **2) 
        elif cfg["voxel_network"]["local_embedding_type"] == "ROIPooling":
            dec_inp_dim += self.voxel_cfg["rgb_out"] * (1 + (cfg["DATASET"]["INPUT_RESOLUTION"][0] // self.voxel_cfg["global_ratio"]) **2)
        elif cfg["voxel_network"]["local_embedding_type"] == "NOROI":
            dec_inp_dim += self.voxel_cfg["rgb_out"] * ((cfg["DATASET"]["INPUT_RESOLUTION"][0] // self.voxel_cfg["global_ratio"]) **2)
      
        # offset decoder
        #print("dec_inp_dim", dec_inp_dim)
        if self.refine_voxel_cfg["refine_offdec_type"] == 'IMNET':
            self.offset_dec = IMNet(inp_dim=dec_inp_dim, out_dim=1, 
                                    gf_dim=self.refine_voxel_cfg["refine_imnet_gf"], use_sigmoid=self.refine_voxel_cfg["refine_use_sigmoid"]).to(self.device)
        elif self.refine_voxel_cfg["refine_offdec_type"] == 'IEF':
            self.offset_dec = IEF(self.device, inp_dim=dec_inp_dim, out_dim=1, gf_dim=self.refine_voxel_cfg["refine_imnet_gf"], 
                                    n_iter=self.refine_voxel_cfg["refine_n_iter"], use_sigmoid=self.refine_voxel_cfg["refine_use_sigmoid"]).to(self.device)
        else:
            raise NotImplementedError('Does not support Offset Decoder Type: {}'.format(self.refine_voxel_cfg["refine_offdec_type"]))
      
    def forward(self, data_dict, loss_dict, cur_iter, refine_forward_times, epoch, mode, refine_hard_neg, refine_hard_neg_ratio):
        if cur_iter == 0:
            self.pred_pos_refine = get_pred_refine(data_dict, data_dict['pred_pos'], mode, cur_iter, self.device, 
                                    self.embeddirs_fn, self.pnet_model, self.embed_fn, self.offset_dec,
                                    local_embedding_type=self.cfg["voxel_network"]["local_embedding_type"], raw_input_type=self.cfg["refine_voxel_network"]["raw_input_type"])
        else:
            self.pred_pos_refine = get_pred_refine(data_dict, self.pred_pos_refine, mode, cur_iter, self.device, 
                                    self.embeddirs_fn, self.pnet_model, self.embed_fn, self.offset_dec,
                                    local_embedding_type=self.cfg["voxel_network"]["local_embedding_type"], raw_input_type=self.cfg["refine_voxel_network"]["raw_input_type"])
        
        if cur_iter == refine_forward_times - 1:
            data_dict['pred_pos'] = self.pred_pos_refine
            loss_dict_refine = compute_refine_loss(data_dict, mode, epoch, refine_hard_neg=refine_hard_neg, refine_hard_neg_ratio=refine_hard_neg_ratio)
            loss_dict_refine['prob_loss'] = torch.zeros(1).to(self.device)
            if mode == "test":
                loss_dict_refine['acc'] = loss_dict['acc']
        
            return loss_dict_refine, data_dict
        else:
            return None, data_dict
        
        


def init_voxel_optimizer(model, refine_model, cfg):
    optim_cfg = cfg["OPTIMIZER"]
    #xyz_params = [param for name, param in model.named_parameters() if 'resnet_woff' in name]
    voxel_params = [param for name, param in model.named_parameters()]
    refine_params = [param for name, param in refine_model.named_parameters()]
    #xyz_optimizer = torch.optim.AdamW([{'params': xyz_params, 'lr': optim_cfg["XYZ_LR"]}])
    refine_optimizer = torch.optim.AdamW([{'params': refine_params, 'lr': optim_cfg["REFINE_LR"]}])
    voxel_optimizer = torch.optim.AdamW([{'params': voxel_params, 'lr': optim_cfg["VOXEL_LR"]}])
    scheduler = MultiStepLR(voxel_optimizer, optim_cfg["DECAY_STEPS"], gamma=0.1)
    return voxel_optimizer, refine_optimizer, scheduler

def load_voxel_model(model, weights_dir, device):
    # weights_dir : xxxx/model.pth

    print(f'restoring checkpoint {weights_dir}')

    checkpoint = torch.load(weights_dir, map_location=device)
    #checkpoint = torch.load(weights_dir)
    print(checkpoint.keys())

    start_epoch = checkpoint["epoch"]
    if "global_iter" in checkpoint:
        global_iter = checkpoint["global_iter"]
    
    state_dict = {}
    for k in checkpoint["model"]:
        #print("k", k)
        if k[7:] in model.state_dict():
            state_dict[k[7:]] = checkpoint["model"][k]
        if k[17:] in model.state_dict():
            state_dict[k[17:]] = checkpoint["model"][k]
    #ret = model.load_state_dict(checkpoint["model"], strict=False)
    ret = model.load_state_dict(state_dict, strict=True)
    print(f'restored "{weights_dir}" model. Key errors:')
    print(ret)
    
#    try:
#        voxel_optimizer.load_state_dict(checkpoint["voxel_optimizer"])
#    except:
#        pass
#    print(f'restore AdamW voxel_optimizer')
#    
#    scheduler.load_state_dict(checkpoint["scheduler"])
#    print(f'restore AdamW scheduler')
    return model, start_epoch, global_iter

def load_refine_model(model, weights_dir, device):
    # weights_dir : xxxx/model.pth

    print(f'restoring checkpoint {weights_dir}')

    checkpoint = torch.load(weights_dir, map_location=device)
    #checkpoint = torch.load(weights_dir)
    print(checkpoint.keys())

    start_epoch = checkpoint["epoch"]
    if "global_iter" in checkpoint:
        global_iter = checkpoint["global_iter"]
    
    state_dict = {}
    for k in checkpoint["model"]:
        #print("k", k)
        if k[7:] in model.state_dict():
            state_dict[k[7:]] = checkpoint["model"][k]
        if k[17:] in model.state_dict():
            state_dict[k[17:]] = checkpoint["model"][k]
    #ret = model.load_state_dict(checkpoint["model"], strict=False)
    ret = model.load_state_dict(state_dict, strict=True)
    print(f'restored "{weights_dir}" model. Key errors:')
    print(ret)
    
#    try:
#        voxel_optimizer.load_state_dict(checkpoint["voxel_optimizer"])
#    except:
#        pass
#    print(f'restore AdamW voxel_optimizer')
#    
#    scheduler.load_state_dict(checkpoint["scheduler"])
#    print(f'restore AdamW scheduler')
    return model, start_epoch, global_iter

def load_optimizer(voxel_optimizer, scheduler,weights_dir, device):
    checkpoint = torch.load(weights_dir, map_location=device)
    try:
        voxel_optimizer.load_state_dict(checkpoint["voxel_optimizer"])
    except:
        pass
    print(f'restore AdamW voxel_optimizer')
    
    scheduler.load_state_dict(checkpoint["scheduler"])
    return voxel_optimizer, scheduler

    

def load_simplenet_model(simplenet_model, path, device):
    state_dict = {}
    simplenet_checkpoint = torch.load(path, map_location=device)["model"]
    for key, value in simplenet_checkpoint.items():
        if key[:6] == "module":
            new_key = "module.simplenet." + key[6:]
        elif "module" not in key:
            new_key = "module.simplenet." + key
        else:
            raise ValueError
        state_dict[new_key] = value
    simplenet_model.load_state_dict(state_dict, strict=True)
    print(f'restored "{path}" model. Key errors:')
    return simplenet_model

def save_weights(save_dir, epoch, global_iter, model, voxel_optimizer,scheduler, cfg):
    save_dict = {
        'epoch': epoch,
        'global_iter': global_iter,
        'model': model.state_dict(),
        'voxel_optimizer': voxel_optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': cfg
    }

    torch.save(save_dict, str(save_dir))

if __name__ == "__main__":
#    resnet_model=resnet_dilated.Resnet34_8s(inp_ch=3, out_ch=32).cuda() 
#    input_img = torch.ones(1, 384, 384, 3).permute(0, 3, 1, 2).cuda() * 255
#    for i in range(1000):
#        t1 = time.time()
#        res = resnet_model(input_img)
#        t2 = time.time()
#        print("t2 - t1", t2 - t1)
    #print(ptn_utils)
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    net = Pointnet2ClsMSG(0).cuda()
    for i in range(50):
        t1 = time.time()
        #net = Pointnet2ClsMSG(0).cuda()
        pts = torch.randn(1, 2048, 3).cuda()
        #print(torch.mean(pts, dim=1))
        pre = net(pts)
        torch.cuda.synchronize()
        t2 = time.time()
        print("t2 - t1", t2 - t1)
    print(pre.shape)
    
    
    
    
    
    
    
    
    
    
    
    


