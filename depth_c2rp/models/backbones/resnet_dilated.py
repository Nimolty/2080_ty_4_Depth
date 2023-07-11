import numpy as np
import torch
import torch.nn as nn
import depth_c2rp.models.backbones.resnet_adjusted as resnet_adjusted 
import time
 

def adjust_input_image_size_for_proper_feature_alignment(input_img_batch, output_stride=8):
    """Resizes the input image to allow proper feature alignment during the
    forward propagation.

    Resizes the input image to a closest multiple of `output_stride` + 1.
    This allows the proper alignment of features.
    To get more details, read here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159

    Parameters
    ----------
    input_img_batch : torch.Tensor
        Tensor containing a single input image of size (1, 3, h, w)

    output_stride : int
        Output stride of the network where the input image batch
        will be fed.

    Returns
    -------
    input_img_batch_new_size : torch.Tensor
        Resized input image batch tensor
    """

    input_spatial_dims = np.asarray( input_img_batch.shape[2:], dtype=np.float )

    # Comments about proper alignment can be found here
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159
    new_spatial_dims = np.ceil(input_spatial_dims / output_stride).astype(np.int) * output_stride + 1

    # Converting the numpy to list, torch.nn.functional.upsample_bilinear accepts
    # size in the list representation.
    new_spatial_dims = list(new_spatial_dims)

    input_img_batch_new_size = nn.functional.upsample_bilinear(input=input_img_batch,
                                                               size=new_spatial_dims)

    return input_img_batch_new_size



class Resnet101_8s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet101_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet101_8s = resnet_adjusted.resnet101(fully_conv=True,
                                        pretrained=True,
                                        output_stride=8,
                                        remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet101_8s.fc = nn.Conv2d(resnet101_8s.inplanes, out_ch, 1)
        
        self.resnet101_8s = resnet101_8s
        
        self._normal_initialization(self.resnet101_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet101_8s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x
    

    
class Resnet18_8s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet18_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet_adjusted.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, out_ch, 1)
        
        self.resnet18_8s = resnet18_8s
        
        self._normal_initialization(self.resnet18_8s.fc)
                
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x, feature_alignment=False):
        
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)
        
        x = self.resnet18_8s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        #x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')#, align_corners=False)
        
        return x

    
class Resnet18_16s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet18_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet18_16s = resnet_adjusted.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=16,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_16s.fc = nn.Conv2d(resnet18_16s.inplanes, out_ch, 1)
        
        self.resnet18_16s = resnet18_16s
        
        self._normal_initialization(self.resnet18_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_16s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x
    

class Resnet18_32s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet18_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_32s = resnet_adjusted.resnet18(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_32s.fc = nn.Conv2d(resnet18_32s.inplanes, out_ch, 1)
        
        self.resnet18_32s = resnet18_32s
        
        self._normal_initialization(self.resnet18_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_32s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x
    

    
class Resnet34_32s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet34_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_32s = resnet_adjusted.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_32s.fc = nn.Conv2d(resnet34_32s.inplanes, out_ch, 1)
        
        self.resnet34_32s = resnet34_32s
        
        self._normal_initialization(self.resnet34_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_32s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x

    
class Resnet34_16s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet34_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_16s = resnet_adjusted.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_16s.fc = nn.Conv2d(resnet34_16s.inplanes, out_ch, 1)
        
        self.resnet34_16s = resnet34_16s
        
        self._normal_initialization(self.resnet34_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_16s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class Resnet34_8s(nn.Module):
    
    
    def __init__(self, inp_ch=4, out_ch=1, global_ratio=64, mid_channels=256, camera_intrin_aware=False):
        
        super(Resnet34_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet_adjusted.resnet34(fully_conv=True,
                                       pretrained=False,
                                       inp_ch=inp_ch,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, out_ch, 1)
        
        self.resnet34_8s = resnet34_8s
        
        self._normal_initialization(self.resnet34_8s.fc)
        self.maxpool = nn.MaxPool2d(global_ratio)
        self.camera_intrin_aware = camera_intrin_aware
        if camera_intrin_aware:
            self.rgb_bn = nn.BatchNorm1d(4)
            self.rgb_mlp = Mlp(4, mid_channels, out_ch)
            self.rgb_se = SELayer(out_ch)  # NOTE: add camera-aware
        
        
        
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x, feature_alignment=False, camera_intrin=False):
        torch.cuda.synchronize()
        tt = time.time()
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)
        
        x = self.resnet34_8s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear', align_corners=False)
        
        if camera_intrin is not False and self.camera_intrin_aware:
            # camera_intrin.shape : B x 4 x 4
            camera_intrin_input = torch.cat([camera_intrin[:, 0, 0:1],
                                             camera_intrin[:, 1, 1:2],
                                             camera_intrin[:, 0, 2:3],
                                             camera_intrin[:, 1, 2:3]
                                            ], dim=-1)
            camera_intrin_input = self.rgb_bn(camera_intrin_input)
            camera_intrin_input = self.rgb_mlp(camera_intrin_input)[..., None, None]
            x = self.rgb_se(x, camera_intrin_input) 

        global_feat = self.maxpool(x)
        
        torch.cuda.synchronize()
        end_time = time.time()
        #print("tt", end_time - tt)
        
        return x, global_feat.flatten(1)
    
class Resnet50_32s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet50_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = resnet_adjusted.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_32s.fc = nn.Conv2d(resnet50_32s.inplanes, out_ch, 1)
        
        self.resnet50_32s = resnet50_32s
        
        self._normal_initialization(self.resnet50_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x

    
class Resnet50_16s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet50_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = resnet_adjusted.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, out_ch, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x

class Resnet50_8s(nn.Module):
    
    
    def __init__(self, out_ch=1000):
        
        super(Resnet50_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = resnet_adjusted.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, out_ch, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x

    

class Resnet9_8s(nn.Module):
    
    # Gets ~ 46 MIOU on Pascal Voc
    
    def __init__(self, out_ch=1000):
        
        super(Resnet9_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet_adjusted.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, out_ch, 1)
        
        self.resnet18_8s = resnet18_8s
        
        self._normal_initialization(self.resnet18_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_8s.conv1(x)
        x = self.resnet18_8s.bn1(x)
        x = self.resnet18_8s.relu(x)
        x = self.resnet18_8s.maxpool(x)

        x = self.resnet18_8s.layer1[0](x)
        x = self.resnet18_8s.layer2[0](x)
        x = self.resnet18_8s.layer3[0](x)
        x = self.resnet18_8s.layer4[0](x)
        
        x = self.resnet18_8s.fc(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x

if __name__ == "__main__":
    resnet_model=Resnet34_8s(inp_ch=3, out_ch=32).cuda() 
    input_img = torch.ones(1, 384, 384,3 ).cuda() * 255
    input_img2 = torch.randn(1, 3, 384, 384).cuda() * 255
    for i in range(1000):
        #t1 = time.time()
        res = resnet_model(input_img.permute(0, 3, 1, 2))
        res = resnet_model(input_img2)
        #print(res)
        #print(time.time() - t1)