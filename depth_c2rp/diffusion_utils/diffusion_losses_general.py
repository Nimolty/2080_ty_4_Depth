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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from depth_c2rp.diffusion_utils import diffusion_utils_general as mutils
from depth_c2rp.diffusion_utils.diffusion_sde_lib_general import VESDE, VPSDE, subVPSDE
#from . import utils as mutils
#from .sde_lib import VESDE, VPSDE


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config["DIFF_OPTIM"]["OPTIMIZER"] == 'Adam':
    # default decay=0, lr=2e-4, beta1=0.9, same as my setting
    optimizer = optim.Adam(params, lr=float(config["DIFF_OPTIM"]["LR"]), betas=(config["DIFF_OPTIM"]["BETA1"], 0.999), eps=float(config["DIFF_OPTIM"]["EPS"]),
                           weight_decay=float(config["DIFF_OPTIM"]["WEIGHT_DECAY"]))  
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def get_simple_optimizer(config, params):
    if config["DIFF_OPTIM"]["OPTIMIZER"] == 'Adam':
        optimizer = optim.Adam(params, lr=float(config["DIFF_OPTIM"]["LR"]))  
    else:
        raise NotImplementedError
    return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config["DIFF_OPTIM"]["LR"],
                  warmup=config["DIFF_OPTIM"]["WARMUP"],
                  grad_clip=config["DIFF_OPTIM"]["GRAD_CLIP"]):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=False, continuous=True, likelihood_weighting=False, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch, mask,t_start=None, t_end=None, train_flag=False):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: [B, j, 3]
      mask: None or same shape as condition
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train_flag, continuous=continuous)
    #print("train_flag", train_flag)
    # prior t0 --> sde.T
#    print("t_start", t_start)
#    print("t_end", t_end)
#    print("model.training", model.training)
    if t_start is not None and t_end is not None:
        print("!!!!!!!")
        print("sde.T", sde.T)
        print("eps", eps)
        print("t_start", t_start)
        print("t_end", t_end)
        t = torch.rand(batch.shape[0], device=batch.device) * (t_end - t_start) + t_start
    else:
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps  # [B]
    
    z = torch.randn_like(batch)  # [B, j, 3]
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None] * z  # [B, j, 3]
    score = score_fn(perturbed_data, t, mask)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None] + z)  # [B, j, 3]
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)  # [B]
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch, mask):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels, mask)
    target = -noise / (sigmas ** 2)[:, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert (isinstance(vpsde, VPSDE) or isinstance(vpsde, subVPSDE)), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch, mask):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None] * noise
    score = model_fn(perturbed_data, labels, mask)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, reduce_mean=False, continuous=True, likelihood_weighting=False, eps=1e-5):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    print("sde loss")
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting, eps=eps)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      print("ddpm!!!")
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, subVPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch, mask=None, t_start=None, t_end=None, train_flag=False):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    #print("model.training", model.training)
    #print("train_flag", train_flag)
    if train_flag:
#      optimizer = state['optimizer']
#      optimizer.zero_grad()
#      loss.backward()
#      optimize_fn(optimizer, model.parameters(), step=state['step'])
#      state['step'] += 1
#      state['ema'].update(model.parameters())
      
      loss = loss_fn(model, batch, mask, train_flag=train_flag)
      
    else:
      with torch.no_grad():
        print("valid")
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch, mask,t_start, t_end, train_flag=train_flag)
        ema.restore(model.parameters())

    return loss

  return step_fn