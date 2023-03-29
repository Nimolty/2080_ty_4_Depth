import torch
from depth_c2rp.models.heads import *
from depth_c2rp.models.backbones import *
from depth_c2rp.models.layers import *
from depth_c2rp.utils.utils import load_pretrained
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter, Gather


head_names = {"FaPN": FaPNHead}
backbone_names = {"ResNet" : ResNet, "ResT" : ResT, "ConvNeXt" : ConvNeXt, "PoolFormer" : PoolFormer}

class build_network(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def init_pretrained(self, pretrained):
        if pretrained:
            # self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
            load_pretrained(self.backbone, pretrained)
            print('pretrain done!')
    
    def forward(self, _input):
        # _input的shape为BxCxHxW
        B, C, H, W = _input.shape
        out = self.backbone(_input)
        mask_out, trans_out, quat_out, joint_pos = self.head(out)
        
#        print('qt shape', quat_out.shape)
#        print('trans shape', trans_out.shape)
#        print('joint_pos shape', joint_pos.shape)
#        print('joint_3d shape', joint_3d.shape)
#        print('before', mask_out.shape)
        
        mask_out = F.interpolate(mask_out, size=[H, W], mode='bilinear', align_corners=False)
        
#        print('after', mask_out.shape)
        return mask_out, trans_out, quat_out, joint_pos
        
def build_model(backbone_all, head_name, model_classes, in_channels, num_joints, output_h, output_w):
    # backbone_name应该形如ResNet_18, ResNet_34诸如此类
    # head_name应该形如FaPN
    # model_classes 是表示最后的分类数量，str
    backbone_name, backbone_index = backbone_all.split("_")
    model_classes = int(model_classes)
    
    # 确认backbone与Head都在里面
    assert backbone_name in backbone_names
    assert head_name in head_names
    
    backbone = backbone_names[backbone_name](backbone_index)
    head = head_names[head_name](in_channels, num_joints, output_h, output_w, model_classes)
    return build_network(backbone, head)




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
    