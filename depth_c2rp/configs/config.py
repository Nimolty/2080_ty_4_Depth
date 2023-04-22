import torch 
import argparse
import yaml
import numpy as np
import os
import time
from tqdm import tqdm

def update_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_resnet18.yaml", help='Configuration file to use')
    parser.add_argument('--exp_id', type=int, default=0, help="Record the Exp ID")
    parser.add_argument('--epoch_id', type=int, default=0, help="Record the Epoch ID")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dr_iter_num", type=int,default=0)
    parser.add_argument("--model_path", type=str,default="")
    parser.add_argument("--syn_test", type=str,default="")
    parser.add_argument("--toy_network",type=str,default="Simple_Net")
    
    parser.add_argument("--three_d_norm", action="store_true")
    parser.add_argument("--three_d_noise_mu1", type=float, default=0.0)
    parser.add_argument("--three_d_noise_mu2", type=float, default=0.0)
    parser.add_argument("--three_d_noise_mu3", type=float, default=0.0)
    parser.add_argument("--three_d_noise_std1", type=float, default=0.0)
    parser.add_argument("--three_d_noise_std2", type=float, default=0.0)
    parser.add_argument("--three_d_noise_std3", type=float, default=0.0)
    parser.add_argument("--three_d_random_drop", type=float, default=0.0)
    
    parser.add_argument("--trained_spdh_net_path", type=str, default="")
    parser.add_argument("--trained_simple_net_path", type=str, default="")
    parser.add_argument("--load_mask", action="store_true")
    parser.add_argument("--load_current_predgt",type=str,default="")
    parser.add_argument("--link_idx", type=str, default="whole")
    
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
     
    #print('cfg.type', type(cfg))
    #print('cfg.keys()', cfg)
    cfg["EXP_ID"] = args.exp_id
    cfg["RESUME"] = args.resume
    cfg["LOCAL_RANK"] = args.local_rank
    cfg["DR_ITER_NUM"] = args.dr_iter_num
    cfg["EPOCH_ID"] = args.epoch_id
    cfg["MODEL_PATH"] = args.model_path
    cfg["SYN_TEST"] = args.syn_test
    cfg["TOY_NETWORK"] = args.toy_network
    
    cfg["THREE_D_NORM"] = args.three_d_norm
    cfg["THREE_D_NOISE_MU1"] = args.three_d_noise_mu1
    cfg["THREE_D_NOISE_MU2"] = args.three_d_noise_mu2
    cfg["THREE_D_NOISE_MU3"] = args.three_d_noise_mu3
    cfg["THREE_D_NOISE_STD1"] = args.three_d_noise_std1
    cfg["THREE_D_NOISE_STD2"] = args.three_d_noise_std2
    cfg["THREE_D_NOISE_STD3"] = args.three_d_noise_std3
    cfg["THREE_D_RANDOM_DROP"] = args.three_d_random_drop
    
    cfg["TRAINED_SPDH_NET_PATH"] = args.trained_spdh_net_path
    cfg["TRAINED_SIMPLE_NET_PATH"] = args.trained_simple_net_path
    cfg["LOAD_MASK"] = args.load_mask
    cfg["LOAD_CURRENT_PREDGT"] = args.load_current_predgt
    cfg["LINK_IDX"] = args.link_idx
    
    return cfg, args



















