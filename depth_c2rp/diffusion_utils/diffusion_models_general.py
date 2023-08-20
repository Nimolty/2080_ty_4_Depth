# from curses import reset_shell_mode
import os
from re import I
# from this import d
import time
import math
import copy
import pickle
import argparse
import functools
from collections import deque

import numpy as np
from scipy import integrate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from torch import autograd


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]
        
class ClassifierFreeSampler(nn.Module):
    def __init__(self, model, w):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.w = w  # guidance stength, 0: no guidance, [0, 4.0] in original paper

    def forward(self, batch, t, condition):
        """
        batch: [B, j, 3] or [B, j, 1]
        t: [B, 1]
        condition: [B, j, 2]
        Return: [B, j, 3] or [B, j, 1] same dim as batch
        """
        out = self.model(batch, t, condition)
        # TODO: fine-grained zero-out
        zeros = torch.zeros_like(condition)
        out_uncond = self.model(batch, t, zeros)
        return out + self.w * (out - out_uncond)
        
def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config["DIFF_MODEL"]["SIGMA_MAX"]), np.log(config["DIFF_MODEL"]["SIGMA_MIN"]), config["DIFF_MODEL"]["NUM_SCALES"]))

    return sigmas


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class ScoreModelFC_GENERAL_Adv(nn.Module):
    """
    Independent condition feature projection layers for each block
    """
    def __init__(self, config,
        n_joints=17, cond_n_joints=17, joint_dim=3, hidden_dim=64, embed_dim=32, cond_dim=2,
        n_blocks=2, use_groupnorm=True, num_angles=7):
        super(ScoreModelFC_GENERAL_Adv, self).__init__()

        self.config = config
        self.n_joints = n_joints
        self.num_angles = num_angles
        self.dim_angles = 1 if self.num_angles==7 else 3
        self.cond_n_joints = cond_n_joints
        self.joint_dim = joint_dim
        self.n_blocks = n_blocks
        self.use_groupnorm = use_groupnorm
        
        self.act = nn.SiLU()

        self.pre_dense= nn.Linear(n_joints * joint_dim, hidden_dim)
        self.pre_dense_t = nn.Linear(embed_dim, hidden_dim)
        self.pre_dense_cond = nn.Linear(hidden_dim, hidden_dim)
        self.pre_dense_joints = nn.Linear(hidden_dim, hidden_dim)
        self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden_dim)
        self.dropout = nn.Dropout(p=0.25)
        

        # time embedding
        self.time_embedding_type = self.config["DIFF_MODEL"]["EMBEDDING_TYPE"]
        if self.time_embedding_type == 'fourier':
            self.gauss_proj = GaussianFourierProjection(embed_dim=embed_dim)
        elif self.time_embedding_type == 'positional':
            self.posit_proj = functools.partial(get_timestep_embedding, embedding_dim=embed_dim)
        else:
            assert 0

        self.shared_time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
        )
        self.register_buffer('sigmas', torch.tensor(get_sigmas(config)))

        # conditional embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(self.cond_n_joints * cond_dim, hidden_dim),
            self.act
        )
        self.joints_embed = nn.Sequential(
            nn.Linear(self.num_angles * self.dim_angles, hidden_dim),
            self.act
        )


        for idx in range(n_blocks):
            setattr(self, f'b{idx+1}_dense1', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense1_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense1_cond', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense1_joints', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_gnorm1', nn.GroupNorm(32, num_channels=hidden_dim))

            setattr(self, f'b{idx+1}_dense2', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense2_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense2_cond', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense2_joints', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_gnorm2', nn.GroupNorm(32, num_channels=hidden_dim))

        self.post_dense = nn.Linear(hidden_dim, n_joints * joint_dim)
    
    def get_cond_embed(self, condition):
        if condition.shape[-1] == 3:
            # unified representation of 2D/3D pose
            z_mask = torch.sum(torch.abs(condition[:, :, -1]), dim=-1, keepdims=True) > 0  # [B, 1]
            # condition = batch[:, :, :3] - condition
            condition[:, :, -1] = condition[:, :, -1] * z_mask.float()  # mask the z-axis of 2d poses
        else:
            assert condition.shape[-1] == 2 # [B, j, 2]
            condition = condition # no mask
        
        bs = condition.shape[0]
        condition = condition.view(bs, -1)
        cond = self.cond_embed(condition)  # [B, hidden]
        pre_cond = self.pre_dense_cond(cond)
        cond_list = [cond, pre_cond]
        
        for idx in range(self.n_blocks):
            this_cond = getattr(self, f'b{idx+1}_dense1_cond')(cond)
            cond_list.append(this_cond)
            this_cond = getattr(self, f'b{idx+1}_dense2_cond')(cond)
            cond_list.append(this_cond)
        return cond_list
    
    def get_time_embed(self, t):
        # time embedding
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            temb = self.posit_proj(timesteps)
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        temb = self.shared_time_embed(temb)
        pre_temb = self.pre_dense_t(temb)
        time_list = [temb, pre_temb]
        for idx in range(self.n_blocks):
            this_time_embed = getattr(self, f'b{idx+1}_dense1_t')(temb)
            time_list.append(this_time_embed)
            this_time_embed = getattr(self, f'b{idx+1}_dense2_t')(temb)
            time_list.append(this_time_embed) 
        
        return time_list
    
    def get_joints_embed(self, joints):
        # joints : B x 7
        bs = joints.shape[0]
        joints = joints.view(bs, -1).contiguous()
        joints_embed = self.joints_embed(joints) # [B, hidden]
        pre_joints_embed = self.pre_dense_joints(joints_embed)
        joints_list = [joints_embed, pre_joints_embed]
        for idx in range(self.n_blocks):
            this_joints_embed = getattr(self, f'b{idx+1}_dense1_joints')(joints_embed)
            joints_list.append(this_joints_embed)
            this_joints_embed = getattr(self, f'b{idx+1}_dense2_joints')(joints_embed)
            joints_list.append(this_joints_embed) 
        
        return joints_list

    
    def forward(self, joints_embed_all, cond_embed_all, prob):
        
        def forward_func(batch, t, mask):
            """
            batch: [B, j, 3] or [B, j, 1]
            t: [B]
            condition: [B, j, 2 or 3]
            mask: [B, j, 2 or 3] only used during evaluation
            Return: [B, j, 3] or [B, j, 1] same dim as batch
            """
            time_embed_all = self.get_time_embed(t)
            bs = batch.shape[0]
            batch = batch.view(bs, -1)  # [B, j*3]
    
            h = self.pre_dense(batch)
            
            h += time_embed_all[1]
            h += cond_embed_all[1]
            h += joints_embed_all[1] * prob
        
        
            if self.use_groupnorm:
                h = self.pre_gnorm(h)
            h = self.act(h)
            h = self.dropout(h)
    
    
            for idx in range(self.n_blocks):
                h1 = getattr(self, f'b{idx+1}_dense1')(h)
                h1 += time_embed_all[2*idx+2]
                h1 += cond_embed_all[2*idx+2]
                h1 += joints_embed_all[2*idx+2] * prob
                if self.use_groupnorm:
                    h1 = getattr(self, f'b{idx+1}_gnorm1')(h1)
                h1 = self.act(h1)
                # dropout, maybe
                h1 = self.dropout(h1)
    
                h2 = getattr(self, f'b{idx+1}_dense2')(h1)
                h2 += time_embed_all[2*idx+3]
                h2 += cond_embed_all[2*idx+3]
                h2 += joints_embed_all[2*idx+3] * prob
                if self.use_groupnorm:
                    h2 = getattr(self, f'b{idx+1}_gnorm2')(h2)
                h2 = self.act(h2)
                # dropout, maybe
                h2 = self.dropout(h2)
    
                h = h + h2
    
            res = self.post_dense(h)  # [B, j*3]
            res = res.view(bs, self.n_joints, -1)  # [B, j, 3]
    
            ''' normalize the output '''
            if self.config["DIFF_MODEL"]["SCALE_BY_SIGMA"]:
                used_sigmas = used_sigmas.reshape((bs, 1, 1))
                res = res / used_sigmas
    
            #print("prob", prob)
            return res 
        return forward_func
        
    
    def fix_joints(self, joints, prob):
        joints_embed_all = self.get_joints_embed(joints)
    
        return joints_embed_all, prob
    
    def fix_cond(self, condition):
        cond_embed_all = self.get_cond_embed(condition)
        
        return cond_embed_all
    
        
