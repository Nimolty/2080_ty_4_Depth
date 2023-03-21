import torch 
import argparse
import yaml
import numpy as np
import os
import time
from tqdm import tqdm

def update_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train.yaml", help='Configuration file to use')
    parser.add_argument('--exp_id', type=int, default=0, help="Record the Exp ID")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dr_iter_num", type=int,default=0)
    
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
     
    print('cfg.type', type(cfg))
    print('cfg.keys()', cfg)
    cfg["EXP_ID"] = args.exp_id
    cfg["RESUME"] = args.resume
    cfg["LOCAL_RANK"] = args.local_rank
    cfg["DR_ITER_NUM"] = args.dr_iter_num
    
    return cfg, args



















