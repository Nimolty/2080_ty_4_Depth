import numpy as np
import time

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as tviz_models
from torch.nn.parallel._functions import Scatter, Gather

from depth_c2rp.models.heads import *
from depth_c2rp.models.backbones import *
from depth_c2rp.models.layers import *
from depth_c2rp.utils.utils import load_pretrained
from depth_c2rp.utils.spdh_network_utils import MLP_TOY, compute_rigid_transform, SoftArgmaxPavlo, SpatialSoftArgmax, SpatialSoftArgmax2d
from depth_c2rp.models.layers.toy_layer import TransformerEncoder, TransformerEncoderLayer
from depth_c2rp.utils.spdh_utils import compute_3n_loss_40, compute_3n_loss_39, compute_3n_loss_42
from depth_c2rp.utils.spdh_sac_utils import compute_rede_rt


head_names = {"FaPN": FaPNHead}
backbone_names = {"ResNet" : ResNet, "ResT" : ResT, "ConvNeXt" : ConvNeXt, "PoolFormer" : PoolFormer, 
                  "stacked_hourglass" : HourglassNet, "hrnet" : HRNet, "dreamhourglass_resnet_h" : ResnetSimple, "dreamhourglass_vgg" : DreamHourglass
                 }
simplenet_names = {"Simple_Net" : MLP_TOY}

class build_spdh_train_network(nn.Module):
    def __init__(self, backbone, simplenet, cfg, device=torch.device("cpu")):
        super().__init__()
        self.backbone = backbone
        self.simplenet = simplenet
        self.device=device  
        self.softargmax_uv = SpatialSoftArgmax(False)
        self.softargmax_uz = SpatialSoftArgmax(False)
        self.cfg = cfg
        self.joints_3d_pred = torch.ones(1, self.cfg["DATASET"]["NUM_JOINTS"]//2, 3)
    
    def forward(self, _input, cam_params, joints_1d_gt=None):
#        start = torch.cuda.Event(enable_timing=True)
#        end = torch.cuda.Event(enable_timing=True)
#        
#        
#        start.record()        
        h, w, c, input_K = cam_params["h"], cam_params["w"], cam_params["c"], cam_params["input_K"]
        pre_heatmap_pred = self.backbone(_input)[-1]
        heatmap_pred = (F.interpolate(pre_heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
        
        # heatmap_pred : B x C x H x W
        B, C, H, W = heatmap_pred.shape
        # use softargmax to get xyz
        joints_3d_pred = self.joints_3d_pred.to(input_K.device).repeat(B, 1, 1)
        uv_pred = self.softargmax_uv(heatmap_pred[:, :C//2, :, :]) # B x C//2 x 2        
        z_pred = self.softargmax_uz(heatmap_pred[:, C//2:, :, :])[:,:, 1:2]  # B x C//2 x 1
        
        joints_3d_pred[:, :, :2] = uv_pred
        Z_min, _, dZ = self.cfg["DATASET"]["DEPTH_RANGE"]
        z_pred = ((z_pred * dZ) + Z_min) / 1000
        inv_intrinsic = torch.inverse(input_K).unsqueeze(1).repeat(1,
                                                               C//2,
                                                               1,
                                                               1)
        joints_3d_pred = (inv_intrinsic @ joints_3d_pred[:, :, :, None]).squeeze(-1)
        joints_3d_pred *= z_pred # B x C//2 x 3
        joints_3d_pred_norm = joints_3d_pred - joints_3d_pred[:, :1, :].clone() # Normalization

        # predict joints angle
        joints_angle_pred = self.simplenet(torch.flatten(joints_3d_pred_norm, 1)) # B x 7

        # predict R and T using pose fitting
        all_dof_pred = joints_angle_pred[:, :, None] # B x 7 x 1
#        if joints_1d_gt is not None: 
#            all_dof_pred = joints_1d_gt
        joints_3d_rob_pred = compute_3n_loss_42(all_dof_pred, input_K.device)
        
#        pose_pred = compute_rigid_transform(joints_3d_rob_pred[:, :18, :], joints_3d_pred_norm[:, :18, :])
#        #print(pose_pred)
#        pose_pred_clone = pose_pred.clone()
#        pose_pred_clone[:, :3, 3] = joints_3d_pred[:, 0] + pose_pred[:, :3, 3] 
#        
        
        pose_pred_clone = []
        for b in range(B):
            pose_pred_clone.append(compute_rede_rt(joints_3d_rob_pred[b:b+1, :, :], joints_3d_pred_norm[b:b+1, :, :]))
        pose_pred_clone = torch.cat(pose_pred_clone)
        pose_pred_clone[:, :3, 3] = joints_3d_pred[:, 0] + pose_pred_clone[:, :3, 3]

#        end.record()
#  
#        # Waits for everything to finish running
#        torch.cuda.synchronize()
#
#        print(start.elapsed_time(end))
#        

#        
#        
#        
        return heatmap_pred, all_dof_pred, pose_pred_clone, joints_3d_rob_pred, joints_3d_pred_norm, joints_3d_pred

class build_spdh_test_network(nn.Module):
    def __init__(self, backbone, simplenet, cfg, device=torch.device("cpu")):
        super().__init__()
        self.backbone = backbone
        self.simplenet = simplenet
        self.device=device  
        self.softargmax_uv = SpatialSoftArgmax(False)
        self.softargmax_uz = SpatialSoftArgmax(False)
        self.cfg = cfg
        self.joints_3d_pred = torch.ones(1, self.cfg["DATASET"]["NUM_JOINTS"]//2, 3).to(self.device)
    
    def forward(self, _input, cam_params):      
        h, w, c, input_K = cam_params["h"], cam_params["w"], cam_params["c"], cam_params["input_K"]
        pre_heatmap_pred = self.backbone(_input)[-1]
        heatmap_pred = (F.interpolate(pre_heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
        
        # heatmap_pred : B x C x H x W
        B, C, H, W = heatmap_pred.shape
        # use softargmax to get xyz
        joints_3d_pred = self.joints_3d_pred.repeat(B, 1, 1)
        uv_pred = self.softargmax_uv(heatmap_pred[:, :C//2, :, :]) # B x C//2 x 2        
        z_pred = self.softargmax_uz(heatmap_pred[:, C//2:, :, :])[:,:, 1:2]  # B x C//2 x 1
        
        joints_3d_pred[:, :, :2] = uv_pred
        Z_min, _, dZ = self.cfg["DATASET"]["DEPTH_RANGE"]
        z_pred = ((z_pred * dZ) + Z_min) / 1000
        inv_intrinsic = torch.inverse(input_K).unsqueeze(1).repeat(1,
                                                               C//2,
                                                               1,
                                                               1)
        joints_3d_pred = (inv_intrinsic @ joints_3d_pred[:, :, :, None]).squeeze(-1)
        joints_3d_pred *= z_pred # B x C//2 x 3
        joints_3d_pred_norm = joints_3d_pred - joints_3d_pred[:, :1, :].clone() # Normalization

        # predict joints angle
        joints_angle_pred = self.simplenet(torch.flatten(joints_3d_pred_norm, 1)) # B x 7

        # predict R and T using pose fitting
        all_dof_pred = joints_angle_pred[:, :, None] # B x 7 x 1
        joints_3d_rob_pred = compute_3n_loss(all_dof_pred, self.device) # B x num_kps x 3
        
        
        
#        pose_pred = compute_rigid_transform(joints_3d_rob_pred[:, :18, :], joints_3d_pred_norm[:, :18, :])
#        #print(pose_pred)
#        pose_pred_clone = pose_pred.clone()
#        pose_pred_clone[:, :3, 3] = joints_3d_pred[:, 0] + pose_pred[:, :3, 3]

        # dsacstar for inference
        dsacstar_cfg = self.cfg["DSACSTAR"]
        joints_3d_rob_pred_clone = joints_3d_rob_pred.clone().detach().cpu()
        joints_3d_rob_pred_clone = joints_3d_rob_pred_clone.transpose(0, 2, 1).reshape(B, 3, 1, -1)
        joints_3d_pred_norm_clone = joints_3d_pred_norm_clone.clone().detach().cpu()
        joints_3d_pred_norm_clone = joints_3d_pred_norm_clone.transpose(0, 2, 1).reshape(B, 3, 1, -1)
        
        
        pose_pred_clone = torch.zeros((B, 4, 4))
        for b in range(B):
            dsacstar.forward_rgbd(
    				joints_3d_pred_norm_clone,  
    				joints_3d_rob_pred_clone, #contains precalculated camera coordinates
    				pose_pred_clone[b], 
    				dsacstar_cfg["hypotheses"], 
    				dsacstar_cfg["threshold"],
    				dsacstar_cfg["inlieralpha"],
    				dsacstar_cfg["maxpixelerror"])
        
       
        return heatmap_pred, all_dof_pred, pose_pred_clone.to(self.device)

         

class build_network(nn.Module):
    def __init__(self, backbone, head, rot_type="quaternion"):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.rot_type = rot_type
    
    def init_pretrained(self, pretrained):
        if pretrained:
            # self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
            load_pretrained(self.backbone, pretrained)
            print('pretrain done!')
    
    def forward(self, _input):
        # _input的shape为BxCxHxW
        B, C, H, W = _input.shape
        out = self.backbone(_input)
        if self.rot_type == "quaternion":
            mask_out, trans_out, quat_out, joint_pos = self.head(out)
            mask_out = F.interpolate(mask_out, size=[H, W], mode='bilinear', align_corners=False)
            return mask_out, trans_out, quat_out, joint_pos
        elif self.rot_type == "o6d":
            mask_out, poses_out, joint_pos = self.head(out)
            mask_out = F.interpolate(mask_out, size=[H, W], mode='bilinear', align_corners=False)
            return mask_out, poses_out, joint_pos
        else:
            raise ValueError

def build_model(backbone_all, head_name, model_classes, in_channels, num_joints, output_h, output_w, rot_type="quaternion"):
    # backbone_name应该形如ResNet_18, ResNet_34诸如此类
    # head_name应该形如FaPN
    # model_classes 是表示最后的分类数量，str
    backbone_name, backbone_index = backbone_all.split("_")
    model_classes = int(model_classes)
    
    # 确认backbone与Head都在里面
    assert backbone_name in backbone_names
    assert head_name in head_names
    
    backbone = backbone_names[backbone_name](backbone_index)
    head = head_names[head_name](in_channels, num_joints, output_h, output_w, model_classes, rot_type=rot_type)
    return build_network(backbone, head, rot_type)

def build_spdh_model(cfg):
    network_name = cfg["MODEL"]["NAME"]
    network_params = {}
    
    # init inchannels
    if cfg["MODEL"]["INPUT_TYPE"] == "D":
        inchannels = 1
    elif cfg["MODEL"]["INPUT_TYPE"] == "XYZ":
        inchannels = 3
    elif cfg["MODEL"]["INPUT_TYPE"] == "RGB":
        inchannels = 3
    elif cfg["MODEL"]["INPUT_TYPE"] == "RGBD":
        inchannels = 4
    else:
        raise ValueError
    
    if network_name == "stacked_hourglass":
        network_params["inchannels"] = inchannels
        network_params["num_stacks"] = cfg["MODEL"]["STACKS"]
        network_params["num_blocks"] = cfg["MODEL"]["BLOCKS"]
        network_params["num_joints"] = cfg["DATASET"]["NUM_JOINTS"]
       
    elif network_name == "hrnet":
        network_params["inchannels"] = inchannels
        network_params["c"] = cfg["MODEL"]["CONV_WIDTH"]
        network_params["num_joints"] = cfg["DATASET"]["NUM_JOINTS"]
        network_params["bn_momentum"] = 0.1
        
    elif "dreamhourglass_resnet" in network_name:
        network_params["n_keypoints"] = cfg["DATASET"]["NUM_JOINTS"]
        network_params["full"] = cfg["MODEL"]["FULL"]
    
    else:
        raise ValueError
    
    model = backbone_names[network_name](**network_params)
    return model
    
def build_whole_spdh_model(cfg, device,mode="train"):
    spdh_net_name = cfg["MODEL"]["NAME"]
    dataset_cfg = cfg["DATASET"]
    spdh_net_params = {}
    simple_net_params = {}
    
    # init inchannels
    if cfg["MODEL"]["INPUT_TYPE"] == "D":
        inchannels = 1
    elif cfg["MODEL"]["INPUT_TYPE"] == "XYZ":
        inchannels = 3
    elif cfg["MODEL"]["INPUT_TYPE"] == "RGB":
        inchannels = 3
    elif cfg["MODEL"]["INPUT_TYPE"] == "RGBD":
        inchannels = 4
    else:
        raise ValueError
    
    if spdh_net_name == "stacked_hourglass":
        spdh_net_params["inchannels"] = inchannels
        spdh_net_params["num_stacks"] = cfg["MODEL"]["STACKS"]
        spdh_net_params["num_blocks"] = cfg["MODEL"]["BLOCKS"]
        spdh_net_params["num_joints"] = cfg["DATASET"]["NUM_JOINTS"]
    elif spdh_net_name == "hrnet":
        spdh_net_params["inchannels"] = inchannels
        spdh_net_params["c"] = cfg["MODEL"]["CONV_WIDTH"]
        spdh_net_params["num_joints"] = cfg["DATASET"]["NUM_JOINTS"]
        spdh_net_params["bn_momentum"] = 0.1
    elif "dreamhourglass_resnet" in spdh_net_name:
        spdh_net_params["n_keypoints"] = cfg["DATASET"]["NUM_JOINTS"]
        spdh_net_params["full"] = cfg["MODEL"]["FULL"]
    else:
        raise ValueError
        
    if cfg["TOY_NETWORK"] == "Simple_Net":
        simple_net_params = {"dim" : dataset_cfg["NUM_JOINTS"] // 2 * 3, "h1_dim" :  1024, "out_dim" : 7}
        
    
    backbone = backbone_names[spdh_net_name](**spdh_net_params)
    simplenet = simplenet_names[cfg["TOY_NETWORK"]](**simple_net_params)
    
    if mode == "train":
        return build_spdh_train_network(backbone, simplenet, cfg, device)
    elif mode == "test":
        return build_spdh_test_network(backbone, simplenet, cfg, device)
    
    

def build_toy_spdh_model(in_dim, hidden_dim, out_dim):
    model = MLP_TOY(in_dim, hidden_dim, out_dim)
    return model
    
def build_mode_spdh_model(kwargs, mode, num_layers=1):
    if mode == "Simple_Net":
        model = MLP_TOY(**kwargs)
    elif mode == "Transformer_Net":
        model = TransformerEncoder(
                TransformerEncoderLayer(**kwargs), num_layers=num_layers
                 )
    return model

def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class _DataParallel(Module):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super(_DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.chunk_sizes = chunk_sizes
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.chunk_sizes)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids, chunk_sizes):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        device_ids: GPU ids on which to replicate module
        output_device: GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Variable containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)

def DataParallel(module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
    if chunk_sizes is None:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
    standard_size = True
    for i in range(1, len(chunk_sizes)):
        if chunk_sizes[i] != chunk_sizes[0]:
            standard_size = False
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(standard_size)
    if standard_size:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
        print('standard_size', standard_size)
    return _DataParallel(module, device_ids, output_device, dim, chunk_sizes)

