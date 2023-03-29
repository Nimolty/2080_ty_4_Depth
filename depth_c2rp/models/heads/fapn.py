import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
import time
from depth_c2rp.models.layers import ConvModule, ConvModule3
from depth_c2rp.models.backbones import ResNet, ResT, ConvNeXt, PoolFormer

class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2,  g* 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2*2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)
    
    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)
        
        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm


class FaPNHead(nn.Module):
    def __init__(self, in_channels, num_joints, output_h, output_w, num_classes=19, channel=128, rot_type="quaternion"): 
        super().__init__()
        in_channels = in_channels[::-1]
        self.num_joints = num_joints
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])
        self.rot_type = rot_type

        for ch in in_channels[1:]:
            self.align_modules.append(FAM(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))
        
        # build mask head
        self.mask_head = nn.Sequential()
        self.mask_head.add_module("0", nn.Conv2d(channel, num_classes, 1))
        self.mask_head.add_module("1", nn.Dropout2d(0.1))
        
        # build quaternion and trans head
        self.qt_common_head = nn.Sequential(*ConvModule3(channel, channel//2), *ConvModule(channel//2, channel//4))
        if self.rot_type == "quaternion":
            self.trans_head = nn.Linear(channel//4 * output_h *output_w, 3) # 这里要改掉的
            self.quat_head = nn.Linear(channel//4 * output_h * output_w, 4)
        elif self.rot_type == "o6d":
            self.o6d_head = nn.Linear(channel//4 * output_h *output_w, 9)
        else:
            raise ValueError
        
        # build 3D joints and pos
        self.joint_common_head = nn.ModuleList([ConvModule(channel, num_joints), nn.Linear(output_h * output_w, channel), nn.ReLU(True)])
        #self.joint_3d_head = nn.Linear(channel, 3)
        self.joint_pos_head = nn.Linear(channel, 1)
        

    def forward(self, features) -> Tensor:
        features = features[::-1]
        
#        for fea in features:
#            print('shape', fea.shape)
        
        out = self.align_modules[0](features[0])
        
        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)
            
#        print('out.shape', out.shape)
        B, _, out_h, out_w = out.shape
        
        mask_out = self.mask_head(out)
        qt_common = self.qt_common_head(out)
        qt_common = qt_common.view(B, -1)
        
        if self.rot_type == "quaternion":
            trans_out = self.trans_head(qt_common)
            quat_out = self.quat_head(qt_common)
        elif self.rot_type == "o6d":
            o6d_out = self.o6d_head(qt_common)
        else:
            raise ValueError

        
        for idx, fc in enumerate(self.joint_common_head):
            if idx == 0:
                joint_out = fc(out)
                joint_out = joint_out.view(B, self.num_joints, -1)
            else:
                joint_out = fc(joint_out)
        
        #joint_3d = self.joint_3d_head(joint_out)
        joint_out = self.joint_pos_head(joint_out)
        
        if self.rot_type == "quaternion":
            return mask_out, trans_out, quat_out, joint_out
        elif self.rot_type == "o6d":
            return mask_out, o6d_out, joint_out
        else:
            return None
        


if __name__ == '__main__':
    # backbone = ResNet('50')
    if torch.cuda.is_available():
        print('Cuda True')
    else:
        print('False')
    
    # backbone = ResNet('18').cuda()
    # head = FaPNHead([64, 128, 256, 512], 11, 100, 100).cuda()

    # backbone = ResNet('50').cuda()
    # head = FaPNHead([256, 512, 1024, 2048], 11, 100, 100).cuda()
    
    # backbone = ResT('S').cuda()
    # head = FaPNHead([64, 128, 256, 512], 11, 100,100).cuda()
    
#    backbone = ResT('B').cuda()
#    head = FaPNHead([96, 192, 384, 768], 11, 100,100).cuda()

#    backbone = ResT('L').cuda()
#    head = FaPNHead([96, 192, 384, 768], 11, 100,100).cuda()

#    backbone = ConvNeXt("B").cuda()
#    head = FaPNHead([128, 256, 512, 1024],11, 100, 100).cuda()
    
    backbone = PoolFormer("M36").cuda()
    head = FaPNHead([96, 192, 384, 768], 11, 100, 100).cuda()
    
    for i in range(300):
        start_time = time.time()
        x = torch.randn(1, 11, 400, 400).cuda()
        end_time = time.time()
        
        print("Image Construction", end_time - start_time) 
        
        time1 = time.time()
        features = backbone(x)
        time2 = time.time()
        out = head(features)
        time3 = time.time()
        print('out.shape', out[1].shape)
        out = F.interpolate(out[0], size=x.shape[-2:], mode='bilinear', align_corners=False)
        time4 = time.time()
        print('backbone time', time2 - time1)    
        print('FaPN head time', time3 - time2) 
        print('Up sampling time', time4 - time3)    
        print('all_time', time4 - time1)
        
    #print(out.shape)