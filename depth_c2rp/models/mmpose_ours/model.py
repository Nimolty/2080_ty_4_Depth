#from mmdet.models.backbones.cspnext import CSPNeXt
from depth_c2rp.models.mmpose_ours.cspnext import CSPNeXt
from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead
import torch
import torch.nn as nn
import numpy as np
import time

from mmpose.registry import KEYPOINT_CODECS, MODELS

input_size = (384, 384)
num_keypoints = 14


class mmpose_network(nn.Module):
    def __init__(self, backbone_params=dict(
                      arch='P5',
                      expand_ratio=0.5,
                      deepen_factor=1.33,
                      widen_factor=1.25,
                      out_indices=(4, ),
                      channel_attention=True,
                      norm_cfg=dict(type='SyncBN'),
                      act_cfg=dict(type='SiLU'),
                      init_cfg=dict(
                          type='Pretrained',
                          prefix='backbone.',
                          checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
                          'rtmposev1/cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth'  # noqa
                       )),
                      head_params=dict(
                            in_channels=1280,
                            out_channels=num_keypoints,
                            input_size=input_size,
                            in_featuremap_size=tuple([s // 32 for s in input_size]),
                            simcc_split_ratio=2.0,
                            final_layer_kernel_size=7,
                            gau_cfg=dict(
                                hidden_dims=256,
                                s=128,
                                expansion_factor=2,
                                dropout_rate=0.,
                                drop_path=0.,
                                act_fn='SiLU',
                                use_rel_bias=False,
                                pos_enc=False),
                            loss=dict(
                                type='KLDiscretLoss',
                                use_target_weight=True,
                                beta=10.,
                                label_softmax=True),
                            decoder=dict(
                                type='SimCCLabel',
                                input_size=input_size,
                                sigma=(6., 6.93),
                                simcc_split_ratio=2.0,
                                normalize=False,
                                use_dark=False)
    
                            ),
                       
                      ):
        super().__init__()
        self.backbone_params = backbone_params
        self.head_params = head_params
        
        self.backbone = CSPNeXt(**self.backbone_params)
        self.head = RTMCCHead(**self.head_params)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        
        return [x]
    
    def predict(self, x, batch_data_samples=[]):
        x = self.backbone(x)
        x = self.head.predict(x, batch_data_samples=[])
        
        x_res = []
        for idx in range(len(x)):
            x_res.append(x[idx].keypoints)
        x_res = np.concatenate(x_res, axis=0)
        return x_res

def build_mmpose_network(name, num_keypoints, input_size, in_channels):
    if name == "rtmpose-m":
        codec = dict(
                type='SimCCLabel',
                input_size=input_size,
                sigma=(6., 6.93),
                simcc_split_ratio=2.0,
                normalize=False,
                use_dark=False)
        backbone_params = dict(
                          in_channels=in_channels,
                          arch='P5',
                          expand_ratio=0.5,
                          deepen_factor=0.67,
                          widen_factor=0.75,
                          out_indices=(4, ),
                          channel_attention=True,
                          norm_cfg=dict(type='SyncBN'),
                          act_cfg=dict(type='SiLU'),
                          init_cfg=dict(
                              type='Pretrained',
                              prefix='backbone.',
                              checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
                              'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
                          ))
        head_params = dict(
                      in_channels=768,
                      out_channels=num_keypoints,
                      input_size=codec['input_size'],
                      in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
                      simcc_split_ratio=codec['simcc_split_ratio'],
                      final_layer_kernel_size=7,
                      gau_cfg=dict(
                          hidden_dims=256,
                          s=128,
                          expansion_factor=2,
                          dropout_rate=0.,
                          drop_path=0.,
                          act_fn='SiLU',
                          use_rel_bias=False,
                          pos_enc=False),
                      loss=dict(
                          type='KLDiscretLoss',
                          use_target_weight=True,
                          beta=10.,
                          label_softmax=True),
                      decoder=codec)
        model = mmpose_network(backbone_params=backbone_params, head_params=head_params)
        return model
    elif name == "rtmpose-l":
        codec = dict(
                type='SimCCLabel',
                input_size=input_size,
                sigma=(6., 6.93),
                simcc_split_ratio=2.0,
                normalize=False,
                use_dark=False)
        backbone_params = dict(
                          in_channels=in_channels,
                          arch='P5',
                          expand_ratio=0.5,
                          deepen_factor=1.,
                          widen_factor=1.,
                          out_indices=(4, ),
                          channel_attention=True,
                          norm_cfg=dict(type='SyncBN'),
                          act_cfg=dict(type='SiLU'),
                          init_cfg=dict(
                              type='Pretrained',
                              prefix='backbone.',
                              checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
                              'rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'  # noqa
                          ))
        head_params = head=dict(
                      in_channels=1024,
                      out_channels=num_keypoints,
                      input_size=codec['input_size'],
                      in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
                      simcc_split_ratio=codec['simcc_split_ratio'],
                      final_layer_kernel_size=7,
                      gau_cfg=dict(
                          hidden_dims=256,
                          s=128,
                          expansion_factor=2,
                          dropout_rate=0.,
                          drop_path=0.,
                          act_fn='SiLU',
                          use_rel_bias=False,
                          pos_enc=False),
                      loss=dict(
                          type='KLDiscretLoss',
                          use_target_weight=True,
                          beta=10.,
                          label_softmax=True),
                      decoder=codec)
        model = mmpose_network(backbone_params=backbone_params, head_params=head_params)
        return model
    elif name == "rtmpose-x":
        codec = dict(
                type='SimCCLabel',
                input_size=input_size,
                sigma=(6., 6.93),
                simcc_split_ratio=2.0,
                normalize=False,
                use_dark=False)
        backbone_params = dict(
                          in_channels=in_channels,
                          arch='P5',
                          expand_ratio=0.5,
                          deepen_factor=1.33,
                          widen_factor=1.25,
                          out_indices=(4, ),
                          channel_attention=True,
                          norm_cfg=dict(type='SyncBN'),
                          act_cfg=dict(type='SiLU'),
                          init_cfg=dict(
                              type='Pretrained',
                              prefix='backbone.',
                              checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
                              'rtmposev1/cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth'  # noqa
                          ))
        head_params = dict(
                      in_channels=1280,
                      out_channels=num_keypoints,
                      input_size=codec['input_size'],
                      in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
                      simcc_split_ratio=codec['simcc_split_ratio'],
                      final_layer_kernel_size=7,
                      gau_cfg=dict(
                          hidden_dims=256,
                          s=128,
                          expansion_factor=2,
                          dropout_rate=0.,
                          drop_path=0.,
                          act_fn='SiLU',
                          use_rel_bias=False,
                          pos_enc=False),
                      loss=dict(
                          type='KLDiscretLoss',
                          use_target_weight=True,
                          beta=10.,
                          label_softmax=True),
                      decoder=codec)
        model = mmpose_network(backbone_params=backbone_params, head_params=head_params)
        return model
    else:
        raise NotImplementedError
#        
#        
#if __name__ == "__main__":
#    model =  mmpose_network().cuda()
#    #model.train()
#    bs = 2
#    x = torch.rand(bs, 3, 384, 384).cuda()
#    
#    loss=dict(
#              type='KLDiscretLoss',
#              use_target_weight=True,
#              beta=10.,
#              label_softmax=True)
#    criterion =  MODELS.build(loss)
#    print(criterion)
#    
#    
#    for i in range(100):
#        t1 = time.time()
#        #output = model.predict(x)
#        pred_x, pred_y = model(x)[0]
#        #print("pred_x", pred_x.shape)
#        #print("pred_y", pred_y.shape)
#        gt_x, gt_y = torch.rand(bs, 14, 1).cuda(), torch.rand(bs, 14, 1).cuda()
#        keypoint_weights = torch.ones(bs, 14, 1).cuda()
#        
#        pred_simcc = (pred_x, pred_y)
#        gt_simcc = (gt_x, gt_y)
#        
#        loss = criterion(pred_simcc, gt_simcc, keypoint_weights)
#        print("loss", loss)
#        t2 = time.time()
        
#        print("t2 - t1", t2 - t1)
#        print("output", output.shape)
#        print("output[0]", output[0].shape)
#        print("output[1]", output[1].shape)   

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        