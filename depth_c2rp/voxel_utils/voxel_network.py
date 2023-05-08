import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import depth_c2rp.models.backbones.resnet_dilated as resnet_dilated
import depth_c2rp.models.backbones.pointnet as pnet
#from depth_c2rp.models.backbones.dream_hourglass import ResnetSimpleWoff

from depth_c2rp.models.heads import *
from depth_c2rp.models.backbones import *
from depth_c2rp.models.layers import *
from depth_c2rp.utils.spdh_network_utils import MLP_TOY

head_names = {"FaPN": FaPNHead}
backbone_names = {"ResNet" : ResNet, "ResT" : ResT, "ConvNeXt" : ConvNeXt, "PoolFormer" : PoolFormer, 
                  "stacked_hourglass" : HourglassNet, "hrnet" : HRNet, "dreamhourglass_resnet_h" : ResnetSimple, "dreamhourglass_vgg" : DreamHourglass,
                  "dreamhourglass_resnet_woff_h": ResnetSimpleWoff,
                 }
simplenet_names = {"Simple_Net" : MLP_TOY}



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
        self.voxel_cfg = cfg["voxel_network"]
        self.device = device
        
        # embedding network
        self.embed_fn, self.embed_ch = get_embedder(self.voxel_cfg["multires"])
        self.embeddirs_fn, self.embeddirs_ch = get_embedder(self.voxel_cfg["multires_views"]) 
        
        # rgb network (here is xyz network)
        #self.resnet_woff = ResnetSimpleWoff(n_keypoints=cfg["DATASET"]["NUM_JOINTS"], n_offs=self.voxel_cfg["n_offs"], full=self.voxel_cfg["full"]).to(self.device)
        self.resnet_model = resnet_dilated.Resnet34_8s(inp_ch=self.voxel_cfg["rgb_in"], out_ch=self.voxel_cfg["rgb_out"], global_ratio=self.voxel_cfg["global_ratio"]) 
        
        # voxel-based pointnet network
        self.pnet_model = pnet.PointNet2Stage(input_channels=self.voxel_cfg["pnet_in"],
                                    output_channels=self.voxel_cfg["pnet_out"], gf_dim=self.voxel_cfg["pnet_gf"])
        self.pnet_model = self.pnet_model.to(self.device)
        
        self.dec_inp_dim = self.voxel_cfg["pnet_out"] + self.voxel_cfg["rgb_out"] * (self.voxel_cfg["roi_out_bbox"]**2 + (cfg["DATASET"]["INPUT_RESOLUTION"][0] // self.voxel_cfg["global_ratio"]) **2) \
                            + 2 * self.embed_ch + self.embeddirs_ch
        
        self.offset_dec = IEF(self.device, inp_dim=self.dec_inp_dim, out_dim=1, gf_dim=self.voxel_cfg["imnet_gf"], 
                                    n_iter=self.voxel_cfg["n_iter"], use_sigmoid=self.voxel_cfg["use_sigmoid"]).to(self.device)
        self.prob_dec = IMNet(inp_dim=self.dec_inp_dim, out_dim=self.voxel_cfg["prob_out_dim"], 
                                gf_dim=self.voxel_cfg["imnet_gf"], use_sigmoid=self.voxel_cfg["use_sigmoid"]).to(self.device)



class build_voxel_refine_network(nn.Module):
    def __init__(self, cfg, device=torch.device("cpu")):
        super().__init__()
        
        # voxel network cfg
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
        
        # decoder input dim
        dec_inp_dim = self.refine_voxel_cfg["refine_pnet_out"] + self.embed_ch + self.embeddirs_ch
        if self.refine_voxel_cfg["refine_rgb_embedding_type"] == 'ROIAlign':
            dec_inp_dim += self.refine_voxel_cfg["refine_rgb_out"] * (self.refine_voxel_cfg["refine_roi_out_bbox"]**2 + (cfg["DATASET"]["INPUT_RESOLUTION"][0] // self.refine_voxel_cfg["global_ratio"]) **2)
        else:
            raise NotImplementedError('Does not support RGB embedding: {}'.format(self.refine_voxel_cfg["refine_rgb_embedding_type"]))
     
        # offset decoder
        if self.refine_voxel_cfg["refine_offdec_type"] == 'IMNET':
            self.offset_dec = IMNet(inp_dim=dec_inp_dim, out_dim=1, 
                                    gf_dim=self.refine_voxel_cfg["refine_imnet_gf"], use_sigmoid=self.refine_voxel_cfg["refine_use_sigmoid"]).to(self.device)
        elif self.refine_voxel_cfg["refine_offdec_type"] == 'IEF':
            self.offset_dec = IEF(self.device, inp_dim=dec_inp_dim, out_dim=1, gf_dim=self.refine_voxel_cfg["refine_imnet_gf"], 
                                    n_iter=self.refine_voxel_cfg["refine_n_iter"], use_sigmoid=self.refine_voxel_cfg["refine_use_sigmoid"]).to(self.device)
        else:
            raise NotImplementedError('Does not support Offset Decoder Type: {}'.format(self.refine_voxel_cfg["refine_offdec_type"]))


def init_voxel_optimizer(model, cfg):
    optim_cfg = cfg["OPTIMIZER"]
    #xyz_params = [param for name, param in model.named_parameters() if 'resnet_woff' in name]
    voxel_params = [param for name, param in model.named_parameters()]
    #xyz_optimizer = torch.optim.AdamW([{'params': xyz_params, 'lr': optim_cfg["XYZ_LR"]}])
    voxel_optimizer = torch.optim.AdamW([{'params': voxel_params, 'lr': optim_cfg["VOXEL_LR"]}])
    scheduler = MultiStepLR(voxel_optimizer, optim_cfg["DECAY_STEPS"], gamma=0.1)
    return voxel_optimizer, scheduler

def load_voxel_model(model, voxel_optimizer, scheduler, weights_dir, device):
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
        if k in model.state_dict():
            state_dict[k] = checkpoint["model"][k]
    #ret = model.load_state_dict(checkpoint["model"], strict=False)
    ret = model.load_state_dict(state_dict, strict=True)
    print(f'restored "{weights_dir}" model. Key errors:')
    print(ret)
    
    try:
        voxel_optimizer.load_state_dict(checkpoint["voxel_optimizer"])
    except:
        pass
    print(f'restore AdamW voxel_optimizer')
    
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(f'restore AdamW scheduler')
    return model, voxel_optimizer, scheduler, start_epoch, global_iter


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
    resnet_model=resnet_dilated.Resnet34_8s(inp_ch=3, out_ch=32).cuda() 
    input_img = torch.ones(1, 384, 384, 3).permute(0, 3, 1, 2).cuda() * 255
    for i in range(1000):
        #t1 = time.time()
        res = resnet_model(input_img)















if __name__ == "__main__":
    
    
    
    
    pass