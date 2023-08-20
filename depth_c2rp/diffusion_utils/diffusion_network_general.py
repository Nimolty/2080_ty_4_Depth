import argparse
from pathlib import Path
import os, sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

# torch related
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')
        
        
from depth_c2rp.diffusion_utils.diffusion_models_general import ScoreModelFC_GENERAL_Adv
from depth_c2rp.configs.config import update_config
from depth_c2rp.diffusion_utils import diffusion_sde_lib_general as sde_lib
from depth_c2rp.diffusion_utils.diffusion_losses_general import optimization_manager, get_step_fn
from depth_c2rp.diffusion_utils import diffusion_sampling_general as sampling
from depth_c2rp.utils.spdh_network_utils import MLP_TOY
from depth_c2rp.utils.spdh_utils import compute_3n_loss_42_rob
from depth_c2rp.utils.spdh_sac_utils import compute_rede_rt

simplenet_names = {"Simple_Net" : MLP_TOY}

def load_simplenet_model(simplenet_model, path, device):
    state_dict = {}
    simplenet_checkpoint = torch.load(path, map_location=device)["model"]
    for key, value in simplenet_checkpoint.items():
        if key[:6] == "module":
            new_key = "module.simplenet." + key[6:]
        elif "module" not in key:
            new_key = "module.simplenet." + key
        else:
            raise ValueError
        state_dict[new_key] = value
    simplenet_model.load_state_dict(state_dict, strict=True)
    #print(f'restored "{path}" model. Key errors:')
    return simplenet_model

def load_single_simplenet_model(simplenet_model, path, device):
    state_dict = {}
    simplenet_checkpoint = torch.load(path, map_location=device)["model"]
    for key, value in simplenet_checkpoint.items():
        if key[:6] == "module":
            new_key = "simplenet." + key[6:]
        elif "module" not in key:
            new_key = "simplenet." + key
        else:
            raise ValueError
        state_dict[new_key] = value
    simplenet_model.load_state_dict(state_dict, strict=True)
    #print(f'restored "{path}" model. Key errors:')
    return simplenet_model

def load_single_heatmap_model(heatmap_model, path, device):
    state_dict = {}
    heatmap_checkpoint = torch.load(path, map_location=device)["model"]
    for key, value in heatmap_checkpoint.items():
        if key[:6] == "module":
            new_key =  key[7:]
        elif "module" not in key:
            new_key = key
        state_dict[new_key] = value
    heatmap_model.load_state_dict(state_dict, strict=True)
    return heatmap_model
    
class build_simple_network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg["DATASET"]
        self.simple_net_params = {"dim" : self.dataset_cfg["NUM_JOINTS"] * 3, "h1_dim" :  1024, "out_dim" : 7}
        self.simplenet = simplenet_names[self.cfg["TOY_NETWORK"]](**self.simple_net_params)
    
    def forward(self, joints_3d_pred, joints_1d_gt=None, gt_angle_flag=False):
        # joints_3d_pred.shape : B x N x 3
        B, n_kps, _ = joints_3d_pred.shape
        joints_3d_pred_norm = joints_3d_pred - joints_3d_pred[:, :1, :].clone() # Normalization
        joints_angle_pred = self.simplenet(torch.flatten(joints_3d_pred_norm, 1)) # B x 7  
        # predict R and T using pose fitting
        all_dof_pred = joints_angle_pred[:, :, None] # B x 7 x 1
        
        if joints_1d_gt is not None and gt_angle_flag: 
            all_dof_pred = joints_1d_gt   
        joints_3d_rob_pred = compute_3n_loss_42_rob(all_dof_pred, joints_3d_pred.device)
        
        pose_pred_clone = []
        for b in range(B):
            pose_pred_clone.append(compute_rede_rt(joints_3d_rob_pred[b:b+1, :, :], joints_3d_pred_norm[b:b+1, :, :]))
        pose_pred_clone = torch.cat(pose_pred_clone)
        pose_pred_clone[:, :3, 3] = joints_3d_pred[:, 0] + pose_pred_clone[:, :3, 3] 
        
        return all_dof_pred, pose_pred_clone

class build_diffusion_network(nn.Module):
    def __init__(self, cfg, device=torch.device("cpu")):
        super().__init__()
        self.cfg = cfg
        self.model_name = self.cfg["DIFF_MODEL"].get("MODEL_NAME", "normal")
        self.device = device
        
        if self.model_name == "normal":
            self.model = ScoreModelFC_Adv(
            self.cfg,
            n_joints=self.cfg["DIFF_MODEL"]["NUM_JOINTS"],
            cond_n_joints=self.cfg["DIFF_MODEL"]["COND_NUM_JOINTS"],
            joint_dim=self.cfg["DIFF_MODEL"]["JOINT_DIM"],
            hidden_dim=self.cfg["DIFF_MODEL"]["HIDDEN_DIM"],
            embed_dim=self.cfg["DIFF_MODEL"]["EMBED_DIM"],
            cond_dim=self.cfg["DIFF_MODEL"]["CONDITION_DIM"],
            use_groupnorm=self.cfg["DIFF_MODEL"]["USE_GROUPNORM"],
            # n_blocks=1,
            )
        elif self.model_name == "general":
            self.model = ScoreModelFC_GENERAL_Adv(
            self.cfg,
            n_joints=self.cfg["DIFF_MODEL"]["NUM_JOINTS"],
            cond_n_joints=self.cfg["DIFF_MODEL"]["COND_NUM_JOINTS"],
            joint_dim=self.cfg["DIFF_MODEL"]["JOINT_DIM"],
            hidden_dim=self.cfg["DIFF_MODEL"]["HIDDEN_DIM"],
            embed_dim=self.cfg["DIFF_MODEL"]["EMBED_DIM"],
            cond_dim=self.cfg["DIFF_MODEL"]["CONDITION_DIM"],
            use_groupnorm=self.cfg["DIFF_MODEL"]["USE_GROUPNORM"],
            num_angles=self.cfg["DIFF_MODEL"].get("NUM_ANGLES", 7),
            # n_blocks=1,
            )
        self.scaler = lambda x: x
        self.inverse_scaler = lambda x: x
        
        # Setup SDEs
        if self.cfg["DIFF_TRAINING"]["SDE"].lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=self.cfg["DIFF_MODEL"]["BETA_MIN"], 
                                        beta_max=self.cfg["DIFF_MODEL"]["BETA_MAX"], 
                                        N=self.cfg["DIFF_MODEL"]["NUM_SCALES"])
            self.sampling_eps = self.cfg["DIFF_SAMPLING"]["SAMPLING_EPS"]
        elif self.cfg["DIFF_TRAINING"]["SDE"].lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=self.cfg["DIFF_MODEL"]["BETA_MIN"], 
                                beta_max=self.cfg["DIFF_MODEL"]["BETA_MAX"], 
                                N=self.cfg["DIFF_MODEL"]["NUM_SCALES"])
            self.sampling_eps = self.cfg["DIFF_SAMPLING"]["SAMPLING_EPS"]
        else:
            sde_name = self.cfg["DIFF_TRAINING"]["SDE"]
            raise NotImplementedError(f"SDE {sde_name} unknown.")
        
        # build trainig and evaluation functions
        self.continuous = self.cfg["DIFF_TRAINING"]["CONTINUOUS"]
        self.reduce_mean = self.cfg["DIFF_TRAINING"]["REDUCE_MEAN"]
        self.likelihood_weighting = self.cfg["DIFF_TRAINING"]["LIKELIHOOD_WEIGHTING"]
        #self.optimize_fn = optimization_manager(self.cfg)
        self.sampling_shape = (self.cfg["DIFF_EVAL"]["BATCH_SIZE"] * self.cfg["DIFF_EVAL"]["NUM_SAMPLES"], self.cfg["DIFF_MODEL"]["NUM_JOINTS"], self.cfg["DIFF_MODEL"]["JOINT_DIM"])
        self.cfg["DIFF_SAMPLING"]["PROBABILITY_FLOW"] = True
        self.sampling_fn = sampling.get_sampling_fn(self.cfg, self.sde, self.sampling_shape, self.inverse_scaler, self.sampling_eps, device=self.device) 
        self.train_step_fn = get_step_fn(self.sde, train=True,
                                       reduce_mean=False, continuous=self.continuous,
                                       likelihood_weighting=self.likelihood_weighting, eps=self.cfg["DIFF_TRAINING"]["EPS"])
        
    def forward(self, batch, condition, optimizer, ema, step, t_start=None, t_end=None, joints=None, prob=None):
        if self.model_name == "normal":
            state = dict(optimizer=optimizer, model=self.model, ema=ema, step=step)
        elif self.model_name == "general":
            assert joints is not None and prob is not None
            joints_embed_all, prob = self.model.fix_joints(joints, prob)
            cond_embed_all = self.model.fix_cond(condition)
            state = dict(optimizer=optimizer, model=self.model(joints_embed_all=joints_embed_all, cond_embed_all=cond_embed_all, prob=prob), ema=ema, step=step)
        cur_loss = self.train_step_fn(state, batch, t_start=t_start, t_end=t_end, train_flag=self.model.training)
        return cur_loss, state
        


if __name__ == "__main__":
    cfg, args = update_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_diffusion_network(cfg).cuda()
