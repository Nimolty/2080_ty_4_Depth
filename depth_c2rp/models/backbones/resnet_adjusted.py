import torch.nn as nn 
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import os
import time
import shutil
import logging
import numpy as np
import torch
import torch.distributed as dist


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def adjust_learning_rate(epoch, optimizer, init_lr, decay_gamma, nepoch_decay):
    lr = init_lr * (decay_gamma ** (epoch // nepoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def reduce_tensor(tensor, reduction='mean'):
    # clone tensor to avoid overwrite issue 
    rt = tensor.clone()
    # sum tensors from all procs and then distribute to all procs
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    if reduction == 'mean':
        return rt / dist.get_world_size()
    elif reduction == 'sum':
        return rt
    else:
        raise ValueError('Reduction type not supported')

def restore(model, state_dict):
    net_state_dict = model.state_dict()
    restore_state_dict = state_dict
    restored_var_names = set()
    print('Restoring:')
    for var_name in restore_state_dict.keys():
        if var_name in net_state_dict:
            var_size = net_state_dict[var_name].size()
            restore_size = restore_state_dict[var_name].size()
            if var_size != restore_size:
                # pass
                print('Shape mismatch for var', var_name, 'expected', var_size, 'got', restore_size)
            else:
                if isinstance(net_state_dict[var_name], torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    net_state_dict[var_name] = restore_state_dict[var_name].data
                try:
                    net_state_dict[var_name].copy_(restore_state_dict[var_name])
                    # print(str(var_name) + ' -> \t' + str(var_size) + ' = ' + str(int(np.prod(var_size) * 4 / 10**6)) + 'MB')
                    restored_var_names.add(var_name)
                except Exception as ex:
                    print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                              var_name, var_size, restore_size))
                    raise ex
    ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
    unset_var_names = sorted(list(set(net_state_dict.keys()) - restored_var_names))
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        print('Did not restore:')
        # print('Did not restore:\n\t' + '\n\t'.join(ignored_var_names))
    if len(unset_var_names) == 0:
        print('No new variables')
    else:
        print('Initialized but did not modify')
        # print('Initialized but did not modify:\n\t' + '\n\t'.join(unset_var_names))


def debug_print(txt, debug=False):
    if debug:
        print(txt)

def create_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_gpu(batch, device):
    for k,v in batch.items():
        if torch.is_tensor(v):
            batch[k] = batch[k].to(device)
    return batch

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    
    kernel_size = np.asarray((3, 3))
    
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2
    
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 inp_ch=3,
                 num_classes=1000,
                 fully_conv=False,
                 remove_avg_pool_layer=False,
                 output_stride=32,
                 additional_blocks=0,
                 multi_grid=(1,1,1) ):
        
        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1
        
        self.remove_avg_pool_layer = remove_avg_pool_layer
        
        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inp_ch, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
        
        self.additional_blocks = additional_blocks
        
        if additional_blocks == 1:
            
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
        
        if additional_blocks == 2:
            
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
        
        if additional_blocks == 3:
            
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer7 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
        
        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
            # In the latest unstable torch 4.0 the tensor.copy_
            # method was changed and doesn't work as it used to be
            #self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    multi_grid=None):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            
            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:
                
                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride
                
            
            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        
        dilation = multi_grid[0] * self.current_dilation if multi_grid else self.current_dilation
            
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
            
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            
            dilation = multi_grid[i] * self.current_dilation if multi_grid else self.current_dilation
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.additional_blocks == 1:
            
            x = self.layer5(x)
        
        if self.additional_blocks == 2:
            
            x = self.layer5(x)
            x = self.layer6(x)
        
        if self.additional_blocks == 3:
            
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
        
        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)
        
        if not self.fully_conv:
            x = x.view(x.size(0), -1)
            
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    
    if pretrained:
        restore(model, model_zoo.load_url(model_urls['resnet18']))
        # if model.additional_blocks:
            
        #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
            
        #     return model
           
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        restore(model, model_zoo.load_url(model_urls['resnet34']))
        
        # if model.additional_blocks:
            
        #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            
        #     return model
           
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        restore(model, model_zoo.load_url(model_urls['resnet50']))
        
        # if model.additional_blocks:
            
        #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
            
        #     return model
           
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    
   
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    
    
    if pretrained:
        restore(model, model_zoo.load_url(model_urls['resnet101']))
        
        # if model.additional_blocks:
            
        #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
            
        #     return model
           
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    
   
    return model
    


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model