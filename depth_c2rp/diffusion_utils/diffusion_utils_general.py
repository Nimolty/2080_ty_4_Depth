# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""

import torch
from depth_c2rp.diffusion_utils import diffusion_sde_lib_general as sde_lib
import numpy as np

def average_quaternion_batch(Q, weights=None):
    """calculate the average quaternion of the multiple quaternions
    Args:
        Q (tensor): [B, num_quaternions, 4]
        weights (tensor, optional): [B, num_quaternions]. Defaults to None.

    Returns:
        oriented_q_avg: average quaternion, [B, 4]
    """
    
    if weights is None:
        weights = torch.ones((Q.shape[0], Q.shape[1]), device=Q.device) / Q.shape[1]
    A = torch.zeros((Q.shape[0], 4, 4), device=Q.device)
    weight_sum = torch.sum(weights, axis=-1)

    oriented_Q = ((Q[:, :, 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("abi,abk->abik", (oriented_Q, oriented_Q))
    A = torch.sum(torch.einsum("abij,ab->abij", (A, weights)), 1)
    A /= weight_sum.reshape(A.shape[0], -1).unsqueeze(-1).repeat(1, 4, 4)

    q_avg = torch.linalg.eigh(A)[1][:, :, -1]
    oriented_q_avg = ((q_avg[:, 0:1] > 0).float() - 0.5) * 2 * q_avg
    return oriented_q_avg


def diff_save_weights(epoch, model, optimizer, ema, step, save_path):
    save_dict = {
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'ema': ema.state_dict(),
              'step': step,
                }
    torch.save(save_dict, save_path)

def diff_load_weights(save_path, device):
    checkpoint = torch.load(save_path, map_location=device)
    return checkpoint

_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)
  score_model = torch.nn.DataParallel(score_model)
  return score_model


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels, mask):
    """Compute the output of the score-based model.

    Args:
      x: [B, j, 3]
      labels: actually timestep(t), should be [B] for my new model
      

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      #print("not train !!!")
      #model.eval()
      return model(x, labels, mask)
    else:
      #model.train()
      return model(x, labels, mask)

  return model_fn


def get_noise_fn(sde, model, mask, train=False, continuous=False):
  """Wraps `noise_fn` so that the model output corresponds to a real time-dependent noise function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A noise prediction function.
  """
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, sde_lib.VPSDE) and continuous:
    def noise_fn(x, t):
      # For VP-trained models, t=0 corresponds to the lowest noise level
      # The maximum value of time embedding is assumed to 999 for
      # continuously-trained models.
      labels = t
#      print("x.shape", x.shape)
#      print("labels.shape", labels.shape)
#      print("mask.shape", mask.shape)
      noise = model_fn(x, labels, mask)
      return noise
  elif isinstance(sde, sde_lib.VPSDE) and (not continuous):
    def noise_fn(x, t):
      labels = t
      noise = model_fn(x, labels, mask)
      return noise
  elif isinstance(sde, sde_lib.subVPSDE) and (not continuous):
    def noise_fn(x, t):
      labels = t
      noise = model_fn(x, labels, mask)
      return noise
  elif isinstance(sde, sde_lib.subVPSDE) and continuous:
    def noise_fn(x, t):
      labels = t 
      noise = model_fn(x, labels, mask)
      std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      return noise
  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return noise_fn

def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)
  #print("continuous", continuous)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, mask):
      """
      x: [B, j, 3]
      t: [B]
      """
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999  # [B]
        
        score = model_fn(x, labels, mask)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)  # [B]
        score = model_fn(x, labels, mask)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.squeeze(-1).long()]
      
      if continuous:
          score = -score / std[:, None, None]  # [B, j, 3]
      else:
          print("score !!!")
          score = score
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, mask):
      """
      x: [B, j, 3]
      t: [B]
      """
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels, mask)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))