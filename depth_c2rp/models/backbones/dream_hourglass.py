# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as tviz_models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#from networks.Dream_hourglass.spatial_softmax import SoftArgmaxPavlo

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This code is adapted from
# https://gitlab-master.nvidia.com/pmolchanov/lpr-3d-hand-pose-rgb-demo/blob/master/handpose/models/image_heatmaps_pose2dZrel_softargmax_slim.py
class SpatialSoftArgmax(nn.Module):
    """
    The spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.

    """

    def __init__(self, normalize=True):
        """Constructor.
        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize
        self.beta = 25.0

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, h, device=device),
                    torch.linspace(-1, 1, w, device=device),
                    #indexing='ij',
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, h, device=device),
                torch.arange(0, w, device=device),
                #indexing='ij',
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
#        torch.cuda.synchronize() 
#        t1 = time.time()
        b, c, h, w = x.shape
        softmax = F.softmax(x.reshape(-1, h * w) * self.beta, dim=-1)
        #print("softmax", softmax)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        yc, xc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)
#        torch.cuda.synchronize() 
#        t2 = time.time()
#        print("t2 - t1", t2 - t1)

        # concatenate and reshape the result
        # to (B, C, 2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c, 2)

class SoftArgmaxPavlo(torch.nn.Module):
    def __init__(self, n_keypoints=5, learned_beta=False, initial_beta=25.0):
        super(SoftArgmaxPavlo, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(7, stride=1, padding=3)
        if learned_beta:
            self.beta = torch.nn.Parameter(torch.ones(n_keypoints) * initial_beta)
        else:
            self.beta = (torch.ones(n_keypoints) * initial_beta).cuda()

    def forward(self, heatmaps, size_mult=1.0):

        epsilon = 1e-8
        bch, ch, n_row, n_col = heatmaps.size()
        n_kpts = ch

        beta = self.beta

        # input has the shape (#bch, n_kpts+1, img_sz[0], img_sz[1])
        # +1 is for the Zrel
        heatmaps2d = heatmaps[:, :n_kpts, :, :]
        heatmaps2d = self.avgpool(heatmaps2d)

        # heatmaps2d has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d.contiguous().view(bch, n_kpts, -1)

        # getting the max value of the maps across each 2D matrix
        map_max = torch.max(heatmaps2d, dim=2, keepdim=True)[0]

        # reducing the max from each map
        # this will make the max value zero and all other values
        # will be negative.
        # max_reduced_maps has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d - map_max

        beta_ = beta.view(1, n_kpts, 1).repeat(1, 1, n_row * n_col)
        # due to applying the beta value, the non-max values will be further
        # pushed towards zero after applying the exp function
        exp_maps = torch.exp(beta_ * heatmaps2d)
        # normalizing the exp_maps by diving it to the sum of elements
        # exp_maps_sum has the shape (#bch, n_kpts, 1)
        exp_maps_sum = torch.sum(exp_maps, dim=2, keepdim=True)
        exp_maps_sum = exp_maps_sum.view(bch, n_kpts, 1, 1)
        normalized_maps = exp_maps.view(bch, n_kpts, n_row, n_col) / (
            exp_maps_sum + epsilon
        )

        col_vals = torch.arange(0, n_col) * size_mult
        col_repeat = col_vals.repeat(n_row, 1)
        col_idx = col_repeat.view(1, 1, n_row, n_col).cuda()
        # col_mat gives a column measurement matrix to be used for getting
        # 'x'. It is a matrix where each row has the sequential values starting
        # from 0 up to n_col-1:
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1

        row_vals = torch.arange(0, n_row).view(n_row, -1) * size_mult
        row_repeat = row_vals.repeat(1, n_col)
        row_idx = row_repeat.view(1, 1, n_row, n_col).cuda()
        # row_mat gives a row measurement matrix to be used for getting 'y'.
        # It is a matrix where each column has the sequential values starting
        # from 0 up to n_row-1:
        # 0,0,0, ..., 0
        # 1,1,1, ..., 1
        # 2,2,2, ..., 2
        # ...
        # n_row-1, ..., n_row-1

        col_idx = Variable(col_idx, requires_grad=False)
        weighted_x = normalized_maps * col_idx.float()
        weighted_x = weighted_x.view(bch, n_kpts, -1)
        x_vals = torch.sum(weighted_x, dim=2)

        row_idx = Variable(row_idx, requires_grad=False)
        weighted_y = normalized_maps * row_idx.float()
        weighted_y = weighted_y.view(bch, n_kpts, -1)
        y_vals = torch.sum(weighted_y, dim=2)

        out = torch.stack((x_vals, y_vals), dim=2)

        return out


class ResnetSimple(nn.Module):
    def __init__(
        self, n_keypoints=7, freeze=False, pretrained=True, full=False,
    ):
        super(ResnetSimple, self).__init__()
        net = tviz_models.resnet101(pretrained=pretrained)
        self.full = full
        self.conv1 = net.conv1
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
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            # This brings it up from 208x208 to 416x416
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.upsample(x)

        if self.full:
            x = self.upsample2(x)

        return [x]

class ResnetSimpleWoff(nn.Module):
    def __init__(
        self, n_keypoints=7, n_offs = 4, freeze=False, pretrained=True, full=False, 
    ):
        super(ResnetSimpleWoff, self).__init__()
        net = tviz_models.resnet101(pretrained=pretrained)
        self.full = full
        #self.conv1 = net.conv1
#        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3) 
#        self.bn1 = net.bn1
#        self.relu = net.relu
#        self.maxpool = net.maxpool
#
#        self.layer1 = net.layer1
#        self.layer2 = net.layer2
#        self.layer3 = net.layer3
#        self.layer4 = net.layer4
        
        self.feat_extractor = nn.Sequential(
                              net.conv1,
                              net.bn1,
                              net.relu,
                              net.maxpool,
                              net.layer1,         
                              net.layer2,
                              net.layer3,
                              net.layer4,
                              )

        # upconvolution and final layer
        BN_MOMENTUM = 0.1
        if not full:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )
            
            self.upsample_woff = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_offs, kernel_size=1, stride=1),
            )
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            # This brings it up from 208x208 to 416x416
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )

    def forward(self, x):

#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        x = self.maxpool(x)
#
#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
#        print(x.shape)
#        torch.cuda.synchronize()
#        t1 = time.time()
        x = self.feat_extractor(x)

        out1 = self.upsample(x.clone())
        out2 = self.upsample_woff(x.clone())

        if self.full:
            x = self.upsample2(x.clone())
        
#        torch.cuda.synchronize()
#        t2 = time.time()
#        print("resnet time", t2 - t1)

        return [out1, out2, x]

class DopeNetworkBelief(nn.Module):
    def __init__(
        self,
        n_keypoints=7,
        include_extractor=True,
        other=0,
        freeze=False,
        pretrained=True,
        feature_extractor="vgg",
        stage_out=6,
    ):
        super(DopeNetworkBelief, self).__init__()

        numBeliefMap = n_keypoints
        numAffinity = 0
        other = 0

        self.feature_extractor = feature_extractor

        # This is the where we decide to get things out
        self.stage_out = stage_out

        if include_extractor:
            if feature_extractor == "vgg":
                temp = tviz_models.vgg19(pretrained=pretrained).features
                self.vgg = nn.Sequential()
                j = 0
                r = 0
                for l in temp:
                    self.vgg.add_module(str(r), l)
                    r += 1
                    if r >= 23:
                        break
                if freeze:
                    for param in self.vgg.parameters():
                        param.requires_grad = False

                self.vgg.add_module(
                    str(r), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
                )
                self.vgg.add_module(str(r + 1), nn.ReLU(inplace=True))
                self.vgg.add_module(
                    str(r + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
                )
                self.vgg.add_module(str(r + 3), nn.ReLU(inplace=True))

            elif feature_extractor == "resnet":
                temp = models.resnet152(pretrained=pretrained)
                vgg = nn.Sequential()
                vgg.add_module("0", temp.conv1)
                vgg.add_module("1", temp.bn1)
                vgg.add_module("2", temp.relu)
                vgg.add_module("3", temp.maxpool)

                t = 4
                for module in temp.layer1:
                    vgg.add_module(str(t), module)
                    t += 1
                for module in temp.layer2:
                    vgg.add_module(str(t), module)
                    t += 1
                if freeze:
                    for param in vgg.parameters():
                        param.requires_grad = False

                # vgg1 = nn.Sequential()
                vgg.add_module(
                    str(t + 1), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
                )
                vgg.add_module(str(t + 2), nn.ReLU(inplace=True))
                vgg.add_module(
                    str(t + 3), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
                )
                vgg.add_module(str(t + 4), nn.ReLU(inplace=True))

                self.vgg = vgg

            # models

        # affinity bit of the model.
        other = 0
        numAffinity = 0

        # 2 is the belief map
        self.m1_2 = self.getModel(128 + other, numBeliefMap, True)
        self.m2_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )
        self.m3_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )
        self.m4_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )
        self.m5_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )
        self.m6_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )

    def forward(self, x):

        out1 = self.vgg(x)
        out1_2 = self.m1_2(out1)

        if self.stage_out == 1:
            return [out1_2]

        out2 = torch.cat([out1_2, out1], 1)
        out2_2 = self.m2_2(out2)

        if self.stage_out == 2:
            return [out1_2, out2_2]

        out3 = torch.cat([out2_2, out1], 1)
        out3_2 = self.m3_2(out3)

        if self.stage_out == 3:
            return [out1_2, out2_2, out3_2]

        out4 = torch.cat([out3_2, out1], 1)
        out4_2 = self.m4_2(out4)

        if self.stage_out == 4:
            return [out1_2, out2_2, out3_2, out4_2]

        out5 = torch.cat([out4_2, out1], 1)
        out5_2 = self.m5_2(out5)

        if self.stage_out == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2]

        out6 = torch.cat([out5_2, out1], 1)
        out6_2 = self.m6_2(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]

    def getModel(self, inChannels, outChannels, first=False):

        model = nn.Sequential()
        kernel = 7
        count = 10
        channels = 128
        padding = 3
        if first:
            padding = 1
            kernel = 3
            count = 6
        # kernel=conv
        model.add_module(
            "0",
            nn.Conv2d(
                inChannels, channels, kernel_size=kernel, stride=1, padding=padding
            ),
        )

        i = 0
        while i < count:
            i += 1
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            if i >= count - 1:
                toAdd = 0
                if first:
                    toAdd = 512 - channels
                model.add_module(
                    str(i),
                    nn.Conv2d(channels, channels + toAdd, kernel_size=1, stride=1),
                )
            else:
                model.add_module(
                    str(i),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        stride=1,
                        padding=padding,
                    ),
                )

        model.add_module(str(i + 1), nn.ReLU(inplace=True))
        model.add_module(
            str(i + 2),
            nn.Conv2d(channels + toAdd, outChannels, kernel_size=1, stride=1),
        )
        # print (model)
        # quit()
        return model


class DreamHourglassMultiStage(nn.Module):
    def __init__(
        self,
        n_keypoints,
        n_image_input_channels=3,
        internalize_spatial_softmax=True,
        learned_beta=True,
        initial_beta=1.0,
        n_stages=2,
        skip_connections=False,
        deconv_decoder=False,
        full_output=False,
    ):
        super(DreamHourglassMultiStage, self).__init__()

        self.n_keypoints = n_keypoints
        self.n_image_input_channels = n_image_input_channels
        self.internalize_spatial_softmax = internalize_spatial_softmax
        self.skip_connections = skip_connections
        self.deconv_decoder = deconv_decoder
        self.full_output = full_output

        if self.internalize_spatial_softmax:
            # This warning is because the forward code just ignores the second head (spatial softmax)
            # Revisit later if we need multistage networks where each stage has multiple output heads that are needed
            print(
                "WARNING: Keypoint softmax output head is currently unused. Prefer training new models of this type with internalize_spatial_softmax = False."
            )
            self.n_output_heads = 2
            self.learned_beta = learned_beta
            self.initial_beta = initial_beta
        else:
            self.n_output_heads = 1
            self.learned_beta = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        assert isinstance(
            n_stages, int
        ), 'Expected "n_stages" to be an integer, but it is {}.'.format(type(n_stages))
        assert (
            0 < n_stages and n_stages <= 6
        ), "DreamHourglassMultiStage can only be constructed with 1 to 6 stages at this time."

        self.num_stages = n_stages

        # Stage 1
        self.stage1 = DreamHourglass(
            n_keypoints,
            n_image_input_channels,
            internalize_spatial_softmax,
            learned_beta,
            initial_beta,
            skip_connections=skip_connections,
            deconv_decoder=deconv_decoder,
            full_output=self.full_output,
        )

        # Stage 2
        if self.num_stages > 1:
            self.stage2 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

        # Stage 3
        if self.num_stages > 2:
            self.stage3 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

        # Stage 4
        if self.num_stages > 3:
            self.stage4 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

        # Stage 5
        if self.num_stages > 4:
            self.stage5 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

        # Stage 6
        if self.num_stages > 5:
            self.stage6 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

    def forward(self, x, verbose=False):

        y_output_stage1 = self.stage1(x)
        y1 = y_output_stage1[0]  # Just keeping belief maps for now
        if self.num_stages == 1:
            return [y1]

        if self.num_stages > 1:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y1_upsampled = y1
            else:
                y1_upsampled = nn.functional.interpolate(
                    y1, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage2 = self.stage2(torch.cat([x, y1_upsampled], dim=1))
            y2 = y_output_stage2[0]  # Just keeping belief maps for now

            if self.num_stages == 2:
                return [y1, y2]

        if self.num_stages > 2:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y2_upsampled = y2
            else:
                y2_upsampled = nn.functional.interpolate(
                    y2, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage3 = self.stage3(torch.cat([x, y2_upsampled], dim=1))
            y3 = y_output_stage3[0]  # Just keeping belief maps for now

            if self.num_stages == 3:
                return [y1, y2, y3]

        if self.num_stages > 3:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y3_upsampled = y3
            else:
                y3_upsampled = nn.functional.interpolate(
                    y3, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage4 = self.stage4(torch.cat([x, y3_upsampled], dim=1))
            y4 = y_output_stage4[0]  # Just keeping belief maps for now

            if self.num_stages == 4:
                return [y1, y2, y3, y4]

        if self.num_stages > 4:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y4_upsampled = y4
            else:
                y4_upsampled = nn.functional.interpolate(
                    y4, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage5 = self.stage5(torch.cat([x, y4_upsampled], dim=1))
            y5 = y_output_stage5[0]  # Just keeping belief maps for now

            if self.num_stages == 5:
                return [y1, y2, y3, y4, y5]

        if self.num_stages > 5:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y5_upsampled = y5
            else:
                y5_upsampled = nn.functional.interpolate(
                    y5, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage6 = self.stage6(torch.cat([x, y5_upsampled], dim=1))
            y6 = y_output_stage6[0]  # Just keeping belief maps for now

            if self.num_stages == 6:
                return [y1, y2, y3, y4, y5, y6]


# Based on DopeHourglassBlockSmall, not using skipped connections
class DreamHourglass(nn.Module):
    def __init__(
        self,
        n_keypoints,
        n_image_input_channels=3,
        internalize_spatial_softmax=True,
        learned_beta=True,
        initial_beta=1.0,
        skip_connections=False,
        deconv_decoder=False,
        full_output=False,
    ):
        super(DreamHourglass, self).__init__()
        self.n_keypoints = n_keypoints
        self.n_image_input_channels = n_image_input_channels
        self.internalize_spatial_softmax = internalize_spatial_softmax
        self.skip_connections = skip_connections
        self.deconv_decoder = deconv_decoder
        self.full_output = full_output

        if self.internalize_spatial_softmax:
            self.n_output_heads = 2
            self.learned_beta = learned_beta
            self.initial_beta = initial_beta
        else:
            self.n_output_heads = 1
            self.learned_beta = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        vgg_t = tviz_models.vgg19(pretrained=True).features

        self.down_sample = nn.MaxPool2d(2)

        self.layer_0_1_down = nn.Sequential()
        self.layer_0_1_down.add_module(
            "0",
            nn.Conv2d(
                self.n_image_input_channels, 64, kernel_size=3, stride=1, padding=1
            ),
        )
        for layer in range(1, 4):
            self.layer_0_1_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_2_down = nn.Sequential()
        for layer in range(5, 9):
            self.layer_0_2_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_3_down = nn.Sequential()
        for layer in range(10, 18):
            self.layer_0_3_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_4_down = nn.Sequential()
        for layer in range(19, 27):
            self.layer_0_4_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_5_down = nn.Sequential()
        for layer in range(28, 36):
            self.layer_0_5_down.add_module(str(layer), vgg_t[layer])

        # Head 1
        if self.deconv_decoder:
            # Decoder primarily uses ConvTranspose2d
            self.deconv_0_4 = nn.Sequential()
            self.deconv_0_4.add_module(
                "0",
                nn.ConvTranspose2d(
                    512,
                    256,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_4.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_4.add_module(
                "2", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_4.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_3 = nn.Sequential()
            self.deconv_0_3.add_module(
                "0",
                nn.ConvTranspose2d(
                    256,
                    128,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_3.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_3.add_module(
                "2", nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_3.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_2 = nn.Sequential()
            self.deconv_0_2.add_module(
                "0",
                nn.ConvTranspose2d(
                    128,
                    64,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_2.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_2.add_module(
                "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_2.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_1 = nn.Sequential()
            self.deconv_0_1.add_module(
                "0",
                nn.ConvTranspose2d(
                    64,
                    64,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_1.add_module("1", nn.ReLU(inplace=True))

        else:
            # Decoder primarily uses Upsampling
            self.upsample_0_4 = nn.Sequential()
            self.upsample_0_4.add_module("0", nn.Upsample(scale_factor=2))

            # should this go before the upsample?
            self.upsample_0_4.add_module(
                "4", nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            )
            self.upsample_0_4.add_module("5", nn.ReLU(inplace=True))
            self.upsample_0_4.add_module(
                "6", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )

            self.upsample_0_3 = nn.Sequential()
            self.upsample_0_3.add_module("0", nn.Upsample(scale_factor=2))
            self.upsample_0_3.add_module(
                "4", nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
            )
            self.upsample_0_3.add_module("5", nn.ReLU(inplace=True))
            self.upsample_0_3.add_module(
                "6", nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            )

            if self.full_output:
                self.upsample_0_2 = nn.Sequential()
                self.upsample_0_2.add_module("0", nn.Upsample(scale_factor=2))
                self.upsample_0_2.add_module(
                    "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_2.add_module("3", nn.ReLU(inplace=True))
                self.upsample_0_2.add_module(
                    "4", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_2.add_module("5", nn.ReLU(inplace=True))

                self.upsample_0_1 = nn.Sequential()
                self.upsample_0_1.add_module("00", nn.Upsample(scale_factor=2))
                self.upsample_0_1.add_module(
                    "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_1.add_module("3", nn.ReLU(inplace=True))
                self.upsample_0_1.add_module(
                    "4", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_1.add_module("5", nn.ReLU(inplace=True))

        # Output head - goes from [batch x 64 x height x width] -> [batch x n_keypoints x height x width]
        self.heads_0 = nn.Sequential()
        self.heads_0.add_module(
            "0", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.heads_0.add_module("1", nn.ReLU(inplace=True))
        self.heads_0.add_module(
            "2", nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        )
        self.heads_0.add_module("3", nn.ReLU(inplace=True))
        self.heads_0.add_module(
            "4", nn.Conv2d(32, self.n_keypoints, kernel_size=3, stride=1, padding=1)
        )

        # Spatial softmax output of belief map
        if self.internalize_spatial_softmax:
            self.softmax = nn.Sequential()
            self.softmax.add_module(
                "0",
                SoftArgmaxPavlo(
                    n_keypoints=self.n_keypoints,
                    learned_beta=self.learned_beta,
                    initial_beta=self.initial_beta,
                ),
            )

    def forward(self, x):

        # Encoder
        x_0_1 = self.layer_0_1_down(x)
        x_0_1_d = self.down_sample(x_0_1)
        x_0_2 = self.layer_0_2_down(x_0_1_d)
        x_0_2_d = self.down_sample(x_0_2)
        x_0_3 = self.layer_0_3_down(x_0_2_d)
        x_0_3_d = self.down_sample(x_0_3)
        x_0_4 = self.layer_0_4_down(x_0_3_d)
        x_0_4_d = self.down_sample(x_0_4)
        x_0_5 = self.layer_0_5_down(x_0_4_d)

        if self.skip_connections:
            decoder_input = x_0_5 + x_0_4_d
        else:
            decoder_input = x_0_5

        # Decoder
        if self.deconv_decoder:
            y_0_5 = self.deconv_0_4(decoder_input)

            if self.skip_connections:
                y_0_4 = self.deconv_0_3(y_0_5 + x_0_3_d)
            else:
                y_0_4 = self.deconv_0_3(y_0_5)

            if self.skip_connections:
                y_0_3 = self.deconv_0_2(y_0_4 + x_0_2_d)
            else:
                y_0_3 = self.deconv_0_2(y_0_4)

            if self.skip_connections:
                y_0_out = self.deconv_0_1(y_0_3 + x_0_1_d)
            else:
                y_0_out = self.deconv_0_1(y_0_3)

            if self.skip_connections:
                output_head_0 = self.heads_0(y_0_out + x_0_1)
            else:
                output_head_0 = self.heads_0(y_0_out)

        else:
            y_0_5 = self.upsample_0_4(decoder_input)

            if self.skip_connections:
                y_0_out = self.upsample_0_3(y_0_5 + x_0_3_d)
            else:
                y_0_out = self.upsample_0_3(y_0_5)

            if self.full_output:
                y_0_out = self.upsample_0_2(y_0_out)
                y_0_out = self.upsample_0_1(y_0_out)

            output_head_0 = self.heads_0(y_0_out)

        # Output heads
        outputs = []
        outputs.append(output_head_0)

        # Spatial softmax output of belief map
        if self.internalize_spatial_softmax:
            softmax_output = self.softmax(output_head_0)
            outputs.append(softmax_output)

        # Return outputs
        return outputs


if __name__ == "__main__":
    import torch

    # Unit test parameters
    n_keypoints = 14
    batch_size = 1
    iterations = 1000

#    # Unit testing network construction and forward
#    print("DreamHourglassMultiStage: upsample decoder")
#    for n in range(1, 7):
#        print("{}-stage".format(n))
#        net = DreamHourglassMultiStage(
#            n_keypoints,
#            n_stages=n,
#            internalize_spatial_softmax=False,
#            deconv_decoder=False,
#            full_output=True,
#        ).cuda()
#        y = net(torch.zeros(batch_size, 3, 400, 400).cuda())
#        print(y[-1].shape)
#        print()
#        del net, y
#        torch.cuda.empty_cache()

    #raise ()

    net_input_height = 384  # 480
    net_input_width = 384  # 640

    net_input = torch.zeros(batch_size, 3, net_input_height, net_input_width)
    print("net_input shape: {}".format(net_input.shape))
    net = ResnetSimpleWoff(n_keypoints, n_offs=4, full=False).cuda()
    for iter in range(iterations):
        torch.cuda.synchronize()
        t1 = time.time()
        y = net(net_input.cuda())
        torch.cuda.synchronize()
        t2 = time.time() 
        print("t2 - t1", t2 - t1)

#    print("ResnetSimple")
#    net = ResnetSimple(n_keypoints).cuda()
#    t1 = time.time()
#    y = net(net_input.cuda())
#    t2 = time.time()
#    print("t2 - t1", t2 - t1)
#    print(y[-1].shape)
#    print()
#    del net, y
#
#    print("ResnetSimpleFull")
#    net = ResnetSimple(n_keypoints, full=True).cuda()
#    y = net(net_input.cuda())
#    print(y[-1].shape)
#    print()
#    del net, y
#
#    print("DOPE")
#    net = DopeNetworkBelief().cuda()
#    y = net(net_input.cuda())
#    print(y[-1].shape)
#    print()
#    del net, y
#
#    torch.cuda.empty_cache()
#
    # Testing DreamHourglass variations
    
    # vgg_q
#    n_keypoints,
#    n_image_input_channels=3,
#    internalize_spatial_softmax=True,
#    learned_beta=True,
#    initial_beta=1.0,
#    skip_connections=False,
#    deconv_decoder=False,
#    full_output=False,
    
    
#    print("########## DreamHourglass vgg_q ##########")
#    net1 = DreamHourglass(
#        n_keypoints,
#        internalize_spatial_softmax=False,
#        skip_connections=False,
#        deconv_decoder=False,
#    ).cuda()
#    for iter in range(iterations):
#        t1 = time.time()
#        y = net1(net_input.cuda())
#        print("shape", y[-1].shape) # [1, 16, 48, 96]
#        t2 = time.time() 
#        print("t2 - t1", t2 - t1)
#    
#    print("########## DreamHourglass vgg_f ##########")
#    net2 = DreamHourglass(
#        n_keypoints,
#        internalize_spatial_softmax=False,
#        skip_connections=False,
#        deconv_decoder=True,
#    ).cuda()
#    for iter in range(iterations):
#        t1 = time.time()
#        y = net2(net_input.cuda())
#        t2 = time.time()
#        print("shape", y[-1].shape) # [1, 16, 192. 384]
#        print("t2 - t1", t2 - t1)
#    
#    
#    print("########## DreamHourglass resnet_h ##########")
#    net3 = ResnetSimple(n_keypoints, full=False).cuda()
#    for iter in range(iterations):
#        t1 = time.time()
#        y = net3(net_input.cuda())
#        t2 = time.time()
#        print("shape", y[-1].shape) # [1, 16, 96, 192]
#        print("t2 - t1", t2 - t1)
#
#    print("########## DreamHourglass resnet_f ##########")
#    net4 = ResnetSimple(n_keypoints, full=True).cuda()
#    for iter in range(iterations):
#        t1 = time.time()
#        y = net4(net_input.cuda())
#        t2 = time.time()
#        print("shape", y[-1].shape) # [1, 16, 192, 384]
#        print("t2 - t1", t2 - t1)







#    net3 = DreamHourglass(
#        n_keypoints,
#        internalize_spatial_softmax=False,
#        skip_connections=False,
#        deconv_decoder=True,
#    ).cuda()
#    y = net3(net_input.cuda())
#    print(y[-1].shape)
#    print()
#    del net3, y
#
#    torch.cuda.empty_cache()
#
#    net5 = DreamHourglass(
#        n_keypoints,
#        internalize_spatial_softmax=False,
#        skip_connections=True,
#        deconv_decoder=False,
#    ).cuda()
#    y = net5(net_input.cuda())
#    print(y[-1].shape)
#    print()
#    del net5, y
#
#    net7 = DreamHourglass(
#        n_keypoints,
#        internalize_spatial_softmax=False,
#        skip_connections=True,
#        deconv_decoder=True,
#    ).cuda()
#    y = net7(net_input.cuda())
#    print(y[-1].shape)
#    print()
#    del net7, y
#
#    torch.cuda.empty_cache()
#
#    # Testing DreamHourglassMultiStage variations
#    print("DreamHourglassMultiStage")
#    for n in range(1, 7):
#        print("{}-stage".format(n))
#        net = DreamHourglassMultiStage(
#            n_keypoints, n_stages=n, internalize_spatial_softmax=False
#        ).cuda()
#        y = net(net_input.cuda())
#        print(y[-1].shape)
#        print()
#        del net, y
#        torch.cuda.empty_cache()
#
#    # Testing DreamHourglassMultiStage variations
#    print("DreamHourglassMultiStage: deconvolutional decoder")
#    for n in range(1, 7):
#        print("{}-stage".format(n))
#        net = DreamHourglassMultiStage(
#            n_keypoints,
#            n_stages=n,
#            internalize_spatial_softmax=False,
#            deconv_decoder=True,
#        ).cuda()
#        y = net(net_input.cuda())
#        print(y[-1].shape)
#        print()
#        del net, y
#        torch.cuda.empty_cache()