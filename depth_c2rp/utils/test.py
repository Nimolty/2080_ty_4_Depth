import os
import sys
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torchvision.models as tviz_models
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class SoftArgmaxPavlo(torch.nn.Module):
    def __init__(self, n_keypoints=5, learned_beta=False, device=torch.device("cpu"), initial_beta=25.0): 
        super(SoftArgmaxPavlo, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(7, stride=1, padding=3)
        self.device = device
        if learned_beta:
            print("beta")
            self.beta = Parameter(torch.ones(n_keypoints) * initial_beta)
            
        else:
            self.beta = (torch.ones(n_keypoints) * initial_beta).to(device)
#        self.one = Parameter(torch.ones(12))
#        self.linear = nn.Linear(20, 20)

    def forward(self, heatmaps, size_mult=1.0):
        
        t1 = time.time()
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
        t2 = time.time()
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
        t3 = time.time()
        t3_mid = time.time()

        col_vals = (torch.arange(0, n_col).view(-1, n_col) * size_mult).to(self.device)
        col_repeat = col_vals.repeat(n_row, 1)
        col_idx = col_repeat.view(1, 1, n_row, n_col)
        
        # col_mat gives a column measurement matrix to be used for getting
        # 'x'. It is a matrix where each row has the sequential values starting
        # from 0 up to n_col-1:
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1
        
        t_ = time.time()
        row_vals = (torch.arange(0, n_row).view(n_row, -1) * size_mult).to(self.device)
        print("123", time.time() - t_)
        row_repeat = row_vals.repeat(1, n_col)
        row_idx = row_repeat.view(1, 1, n_row, n_col)
        
        
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
        t4 = time.time()
        
        print("t4 - t_", t4 - t_)
        print("t_ - t3_mid", t_ - t3_mid)
        print("t3_mid - t3", t3_mid - t3)
        print("t3 - t2", t3 - t2)
        print("t2 - t1", t2 - t1)
        

        return out

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
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w) * self.beta, dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        yc, xc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C, 2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c, 2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softargmax = SoftArgmaxPavlo(n_keypoints=24, device=device).to(device)
    spatialsoftargmax = SpatialSoftArgmax(False)
    a = torch.randn(12, 24, 384, 384).to(device)
    for i in range(10000):
        #a = torch.randn(12, 24, 384, 384).to(device)
        #torch.cuda.synchronize() 
        t1 = time.time()
        spatialsoftargmax(a)
        #a = torch.arange(192).to(device)
        t2 = time.time()
        #torch.cuda.synchronize() 
        print(t2 - t1)