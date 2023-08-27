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
    parser.add_argument('--exp_id', type=str, default="0", help="Record the Exp ID")
    parser.add_argument('--epoch_id', type=int, default=0, help="Record the Epoch ID")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_checkpoint', type=str, default="")
    parser.add_argument('--resume_heatmap', type=str, default="")
    parser.add_argument('--resume_refine', action='store_true')
    parser.add_argument('--resume_simplenet', type=str, default="")
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    
    parser.add_argument("--dr_iter_num", type=int,default=0)
    parser.add_argument("--dr_loss",  type=str, default="mse")
    parser.add_argument("--dr_input", type=str, default="sim")
    parser.add_argument("--dr_order", type=str, default="single")
    
    parser.add_argument("--model_path", type=str,default="")
    parser.add_argument("--syn_test", type=str,default="")
    parser.add_argument('--real_test',type=str,default="")
    parser.add_argument("--toy_network",type=str,default="Simple_Net")
    
    parser.add_argument("--cx_delta", type=int, default=0)
    parser.add_argument("--cy_delta", type=int, default=0)
    parser.add_argument("--change_intrinsic", action="store_true")
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
    parser.add_argument("--pred_mask", action="store_true")
    parser.add_argument("--gt_angle", action="store_true")
    parser.add_argument("--pred_2d", action="store_true")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--load_current_predgt",type=str,default="")
    parser.add_argument("--link_idx", type=str, default="whole")
    parser.add_argument("--link_dict", type=dict)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--eval_bs", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="ddim_solver")
    
    parser.add_argument("--training_loss_start_idx", type=float, default=0.001)
    parser.add_argument("--training_loss_end_idx", type=float, default=1000.0)
    parser.add_argument("--kps_14_name", type=str, default="42")
    parser.add_argument("--infer_prob", type=float, default=0.0)
    parser.add_argument("--sigma_pixel",type=float, default=15.0)
    
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
     
    #print('cfg.type', type(cfg))
    #print('cfg.keys()', cfg)
    cfg["EXP_ID"] = args.exp_id
    cfg["RESUME"] = args.resume
    cfg["RESUME_CHECKPOINT"] = args.resume_checkpoint
    cfg["RESUME_HEATMAP"] = args.resume_heatmap
    cfg["RESUME_REFINE"] = args.resume_refine
    cfg["RESUME_SIMPLENET"] = args.resume_simplenet
    cfg["LOCAL_RANK"] = args.local_rank
    
    cfg["DR_ITER_NUM"] = args.dr_iter_num
    cfg["DR_LOSS"] = args.dr_loss
    cfg["DR_INPUT"] = args.dr_input
    cfg["DR_ORDER"] = args.dr_order
    
    
    cfg["EPOCH_ID"] = args.epoch_id
    cfg["MODEL_PATH"] = args.model_path
    cfg["SYN_TEST"] = args.syn_test
    cfg["TOY_NETWORK"] = args.toy_network
    cfg["PRED_2D_FLAG"] = args.pred_2d
     
    cfg["CX_DELTA"] = args.cx_delta
    cfg["CY_DELTA"] = args.cy_delta
    cfg["CHANGE_INTRINSIC"] = args.change_intrinsic
    cfg["THREE_D_NORM"] = args.three_d_norm
    cfg["THREE_D_NOISE_MU1"] = args.three_d_noise_mu1
    cfg["THREE_D_NOISE_MU2"] = args.three_d_noise_mu2
    cfg["THREE_D_NOISE_MU3"] = args.three_d_noise_mu3
    cfg["THREE_D_NOISE_STD1"] = args.three_d_noise_std1
    cfg["THREE_D_NOISE_STD2"] = args.three_d_noise_std2
    cfg["THREE_D_NOISE_STD3"] = args.three_d_noise_std3
    cfg["THREE_D_RANDOM_DROP"] = args.three_d_random_drop
    cfg["KPS_14_NAME"] = args.kps_14_name
    
    
    cfg["TRAINED_SPDH_NET_PATH"] = args.trained_spdh_net_path
    cfg["TRAINED_SIMPLE_NET_PATH"] = args.trained_simple_net_path
    cfg["LOAD_MASK"] = args.load_mask
    cfg["PRED_MASK"] = args.pred_mask
    cfg["LOAD_CURRENT_PREDGT"] = args.load_current_predgt
    cfg["LINK_IDX"] = args.link_idx
    cfg["LINK_DICT"] = args.link_dict
    cfg["INFER_PROB"] = args.infer_prob
    cfg["SIGMA_PIXEL"] = args.sigma_pixel
    
    if args.real_test != "":
        cfg["DATASET"]["REAL_ROOT"] = args.real_test
    
    if args.syn_test != "":
        cfg["DATASET"]["VAL_ROOT"] = args.syn_test
    
    if args.gt_angle:
        cfg["DIFF_TRAINING"]["GT_ANGLE_FLAG"] = args.gt_angle
    
    if args.eval_bs:
        cfg["DIFF_EVAL"]["BATCH_SIZE"] = args.eval_bs
    
    cfg["DIFF_EVAL"]["NUM_SAMPLES"] = args.num_samples 
    
    if args.pred_2d:
        cfg["DIFF_MODEL"]["INPUT_TYPE"] = "XYZ"
        if args.load_mask:
            cfg["DIFF_MODEL"]["INPUT_TYPE"] = "XYZ MASK"
            if args.pred_mask:
                cfg["DIFF_MODEL"]["INPUT_TYPE"] = "XYZ PRED_MASK" 
    if args.rgb:
        cfg["DIFF_MODEL"]["INPUT_TYPE"] = "RGB"
        if args.load_mask:
            cfg["DIFF_MODEL"]["INPUT_TYPE"] = "RGB MASK"
            if args.pred_mask:
                cfg["DIFF_MODEL"]["INPUT_TYPE"] = "RGB PRED_MASK"             
    
    if args.pred_2d and args.rgb:
        cfg["DIFF_MODEL"]["INPUT_TYPE"] = "RGB"
        if args.load_mask:
            cfg["DIFF_MODEL"]["INPUT_TYPE"] = "RGB MASK"
            if args.pred_mask:
                cfg["DIFF_MODEL"]["INPUT_TYPE"] = "RGB PRED_MASK" 
    
    if "DIFF_SAMPLING" in cfg:
        cfg["DIFF_SAMPLING"]["METHOD"] = args.sampler
    

    cfg["training_loss_start_idx"] = args.training_loss_start_idx
    cfg["training_loss_end_idx"] = args.training_loss_end_idx
    
    
    
    return cfg, args



















