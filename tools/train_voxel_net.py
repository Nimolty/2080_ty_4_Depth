import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.utils.data import (DataLoader, Dataset)
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


# spdh
from depth_c2rp.utils.utils import load_camera_intrinsics, set_random_seed, exists_or_mkdir, visualize_inference_results
from depth_c2rp.configs.config import update_config
from depth_c2rp.utils.spdh_utils import reduce_mean, init_spdh_model
from depth_c2rp.spdh_optimizers import init_optimizer
from depth_c2rp.utils.spdh_visualize import get_blended_images, get_joint_3d_pred, log_and_visualize_single
from depth_c2rp.utils.spdh_network_utils import SequentialDistributedSampler, distributed_concat
from depth_c2rp.utils.analysis import flat_add_from_pose, add_metrics

# voxel
from depth_c2rp.datasets.datasets_voxel_ours import Voxel_dataset
from depth_c2rp.voxel_utils.voxel_network import build_voxel_network, init_voxel_optimizer, load_voxel_model, save_weights
from depth_c2rp.voxel_utils.voxel_batch_utils import prepare_data, get_valid_points, get_occ_vox_bound, get_miss_ray, compute_ray_aabb, compute_gt, get_embedding_ours, get_pred, compute_loss, get_embedding

from depth_c2rp.models.backbones.dream_hourglass import ResnetSimple, SpatialSoftArgmax

#from inference import network_inference
#from inference_spdh_multi import network_inference


def main(cfg):
    """
        This code is designed for the whole training. 
    """
    # expect cfg is a dict
    set_random_seed(int(cfg["MODEL"]["SEED"]))
    
    assert type(cfg) == dict
    device_ids = [4,5,6,7] 
    
    if cfg["LOCAL_RANK"] != -1:
        torch.cuda.set_device(cfg["LOCAL_RANK"])
        device=torch.device("cuda",cfg["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    
    # Build DataLoader
    dataset_cfg, train_cfg = cfg["DATASET"], cfg["TRAIN"]
    model_cfg = cfg["MODEL"]
    training_data_dir = dataset_cfg["TRAINING_ROOT"]
    camera_K = load_camera_intrinsics(os.path.join(training_data_dir, "_camera_settings.json"))
    
    # Build training and validation set
    training_dataset = Voxel_dataset(train_dataset_dir=dataset_cfg["TRAINING_ROOT"],
                                     val_dataset_dir=dataset_cfg["VAL_ROOT"],
                                     real_dataset_dir=dataset_cfg["REAL_ROOT"],
                                     joint_names=[f"panda_joint_3n_{i+1}" for i in range(int(dataset_cfg["NUM_JOINTS"]))],
                                     run=[0],
                                     init_mode="train", 
                                     #noise=self.cfg['noise'],
                                     img_type=dataset_cfg["TYPE"],
                                     raw_img_size=tuple(dataset_cfg["RAW_RESOLUTION"]),
                                     input_img_size=tuple(dataset_cfg["INPUT_RESOLUTION"]),
                                     sigma=dataset_cfg["SIGMA"],
                                     norm_type=dataset_cfg["NORM_TYPE"],
                                     network_input=model_cfg["INPUT_TYPE"],
                                     network_task=model_cfg["TASK"],
                                     depth_range=dataset_cfg["DEPTH_RANGE"],
                                     depth_range_type=dataset_cfg["DEPTH_RANGE_TYPE"],
                                     aug_type=dataset_cfg["AUG_TYPE"],
                                     aug_mode=dataset_cfg["AUG"],
                                     change_intrinsic=dataset_cfg["CHANGE_INTRINSIC"],
                                     uv_input=cfg["voxel_network"]["uv_input"],
                                     )
                                    
    training_dataset.train()
    train_sampler = DistributedSampler(training_dataset)
    training_loader = DataLoader(training_dataset, sampler=train_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True) 
    
    
    val_dataset = copy.copy(training_dataset)  
    val_dataset.eval()    
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    real_dataset = copy.copy(training_dataset)
    real_dataset.real()
    real_sampler = DistributedSampler(real_dataset)
    real_loader = DataLoader(real_dataset, sampler=real_sampler, batch_size=train_cfg["BATCH_SIZE"],
                                  num_workers=train_cfg["NUM_WORKERS"], pin_memory=True, drop_last=False)
    
    # Build Recording and Saving Path
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    save_path = os.path.join(cfg["SAVE_DIR"], str(cfg["EXP_ID"]))
    checkpoint_path = os.path.join(save_path, "CHECKPOINT")
    tb_path = os.path.join(save_path, "TB")
    exists_or_mkdir(save_path)
    exists_or_mkdir(checkpoint_path)
    exists_or_mkdir(tb_path)
    
    if dist.get_rank() == 0:
        writer = SummaryWriter(tb_path)
        
        # log experiment setting
        param_str = ''
        for key, value in cfg.items():
            if isinstance(value, dict):
                for key1, value1 in value.items():
                    param_str += f'{key}.{key1}: {value1}  \n'
            else:
                param_str += f'{key}: {value}  \n'
        writer.add_text('Experiment setting', param_str)

        
        torch.distributed.barrier()
    
    # Build Model 
    model = build_voxel_network(cfg, device)

    start_epoch, global_iter = 0, 0
    model = model.to(device)
    num_gpus = torch.cuda.device_count()
    print("device", device)
    #if num_gpus > 1:
    print('use {} gpus!'.format(num_gpus))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    #model = torch.compile(model, mode="reduce-overhead")
                                                
    heatmap_model = ResnetSimple(
                                n_keypoints=cfg["DATASET"]["NUM_JOINTS"] * 2,
                                full=cfg["voxel_network"]["full"],
                                )
    heatmap_model = heatmap_model.to(device)
    heatmap_model = torch.nn.parallel.DistributedDataParallel(heatmap_model, device_ids=[cfg["LOCAL_RANK"]],
                                                output_device=cfg["LOCAL_RANK"],find_unused_parameters=False)
    
    if cfg["RESUME_HEATMAP"] != "":
        heatmap_model.load_state_dict(torch.load(cfg["RESUME_HEATMAP"], map_location=device)["model"], strict=True)
        print("successfully loading heatmap!")
    
    softargmax_uv = SpatialSoftArgmax(False)
    
    # Build Opimizer
    voxel_optimizer, scheduler = init_voxel_optimizer(model, cfg)
    
    # Load Trained Model
    
    if cfg["RESUME"]:
        print("Resume Model Checkpoint")
        checkpoint_paths = os.listdir(checkpoint_path)
        checkpoint_paths.sort() 
        this_ckpt_path = os.path.join(checkpoint_path, "model.pth")  
        print('this_ckpt_path', this_ckpt_path)
        model, voxel_optimizer, scheduler, start_epoch, global_iter = load_voxel_model(model, voxel_optimizer, scheduler, this_ckpt_path, device)
        print("successfully loaded!")
    
    #heatmap_model = torch.compile(heatmap_model, mode="reduce-overhead")
    
    # set network
    embed_fn = model.module.embed_fn
    embeddirs_fn = model.module.embeddirs_fn
    #resnet_woff = model.module.resnet_woff
    resnet_model = model.module.resnet_model
    pnet_model = model.module.pnet_model
    offset_dec = model.module.offset_dec
    prob_dec = model.module.prob_dec
    
    
    # Build Loss Function
    heatmap_criterion = torch.nn.MSELoss()
    woff_criterion = torch.nn.MSELoss()
    loss_cfg = cfg["LOSS"]
    

    print("len_dataloader", len(training_loader))
    visualize_iteration = len(training_loader) // 2
    
#    with torch.no_grad():
    for epoch in tqdm(range(start_epoch + 1, train_cfg["EPOCHS"] + 1)):        
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(tqdm(training_loader)):
            torch.cuda.synchronize()
            t1 = time.time()
#            if batch_idx >= 1:
#                print("loading time", t1 - prev_time)
        
            curr_loss = 0.0
        
            # heatmap
#            input_tensor = batch['rgb_img'].to(device).float()
            joints_3d = batch['joints_3D_Z'].to(device).float()
#            b, c, h, w = input_tensor.size()
#            uv_ind = batch["uv_ind"].to(device).type(torch.int64)
#            uv_off = batch["uv_off"].to(device).float()
#            
#            outputs = resnet_woff(input_tensor)
#            torch.cuda.synchronize()
#            t2 = time.time() 
#            heatmap_pred, off_pred, full_xyz_feat = outputs[0], outputs[1], outputs[2]
#            full_xyz_feat = F.interpolate(full_xyz_feat, (h, w), mode='bilinear', align_corners=False)
#            #print("full_xyz_feat.shape", full_xyz_feat.shape)
#            heatmap_gt = batch['heatmap_25d'].to(device).float()
#            heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
#            heatmap_loss = loss_cfg["hm_coeff"] * heatmap_criterion(heatmap_gt, heatmap_pred)
#            
#            off_pred = F.interpolate(off_pred, (h, w), mode='bicubic',align_corners=False) # B x 2 x H x W
#            off_pred = off_pred.permute(0, 2, 3, 1).contiguous()
#            pred_uv_off = ((off_pred[:, :, :, :2]).view(b, -1, 2).contiguous())
#            uv_ind = uv_ind[:, :, :1].expand(uv_ind.size(0), uv_ind.size(1), pred_uv_off.size(2))
#            pred_uv_off = pred_uv_off.gather(1, uv_ind)
#            woff_loss = loss_cfg["woff_coeff"] * woff_criterion(pred_uv_off, uv_off) 
                  
            # prepare data
            data_dict = prepare_data(batch)
            
            # get valid pts other than kps
            get_valid_points(data_dict, valid_sample_num=cfg["voxel_network"]["valid_sample_num"], rgb_in=cfg["voxel_network"]["rgb_in"])
            
            # get occupied voxel data
            get_occ_vox_bound(data_dict, res=cfg["voxel_network"]["res"])
            
            # get kps ray data
            get_miss_ray(data_dict)
            
            # ray AABB slab test
            intersect_pair_flag = compute_ray_aabb(data_dict) 
            
            # compute gt
            compute_gt(data_dict)
            
            # get embedding
            #get_embedding_ours(data_dict, embed_fn, embeddirs_fn, full_xyz_feat, pnet_model)
            get_embedding(data_dict, embed_fn, embeddirs_fn, resnet_model, pnet_model, local_embedding_type=cfg["voxel_network"]["local_embedding_type"]) 
            
            # get pred
            get_pred(data_dict, "train", epoch-1, offset_dec, prob_dec, device, raw_input_type=cfg["voxel_network"]["raw_input_type"])
            torch.cuda.synchronize()
            t3 = time.time()
            
            
            # compute loss
            loss_dict = compute_loss(data_dict, "train", epoch,device) 
            pos_loss = loss_cfg["pos_coeff"] * loss_dict['pos_loss'] 
            prob_loss = loss_cfg["prob_coeff"] * loss_dict['prob_loss']
            #print("pos_loss", pos_loss)
            #print("prob_loss", prob_loss)
            
            #curr_loss = heatmap_loss + woff_loss + pos_loss + prob_loss
            curr_loss = pos_loss + prob_loss
            torch.cuda.synchronize()
            #t4 = time.time()
            #print("time", t2 - t1)
            
#            xyz_optimizer.zero_grad()
            voxel_optimizer.zero_grad()
            curr_loss.backward()
#            xyz_optimizer.step() 
            voxel_optimizer.step() 
            
            torch.cuda.synchronize()
            t4 = time.time()
            
#            print("t4 - t3", t4 - t3)
#            print("t3 - t2", t3 - t2)
#            print("t2 - t1", t2 - t1)
            
            
            # log cur_loss 
            torch.distributed.barrier()
            curr_loss = reduce_mean(curr_loss, num_gpus)
            pos_loss = reduce_mean(pos_loss, num_gpus)
            prob_loss = reduce_mean(prob_loss, num_gpus)
            
#            bs, n_kps, _ = joints_3d.shape
#            joints_3d_gt = joints_3d.clone().cpu().numpy()[:8] # 8 x n x 3
#            gt_images = batch['depthvis'].clone().numpy()[:8]
#            K = batch['K_depth'].clone().numpy()[:8]
#            joints_3d_pred = data_dict['pred_pos'].reshape(bs, n_kps, 3).clone().detach().cpu().numpy()[:8]
#            
#            print(joints_3d_pred - joints_3d_gt)
        
            #print("loss", curr_loss)
            
            if dist.get_rank() == 0:
                writer.add_scalar(f'Train/Train Loss', curr_loss.detach().item(), global_iter)
#                writer.add_scalar(f'Train/Heatmap Loss', heatmap_loss.detach().item(), global_iter)
#                writer.add_scalar(f'Train/Woff Loss', woff_loss.detach().item(), global_iter)
                writer.add_scalar(f'Train/Pos Loss', pos_loss.detach().item(), global_iter)
                writer.add_scalar(f'Train/Prob Loss', prob_loss.detach().item(), global_iter)
                #print(optimizer.param_groups)
                #writer.add_scalar(f"Train/xyz_LR", xyz_optimizer.param_groups[0]['lr'], global_iter)
                writer.add_scalar(f"Train/voxel_LR", voxel_optimizer.param_groups[0]['lr'], global_iter)
                   
            if batch_idx % visualize_iteration == 0 and dist.get_rank() == 0:
                bs, n_kps, _ = joints_3d.shape
                joints_3d_gt = joints_3d.clone().cpu().numpy()[:8] # 8 x n x 3
                gt_images = batch['depthvis'].clone().numpy()[:8]
                K = batch['K_depth'].clone().numpy()[:8]
                joints_3d_pred = data_dict['pred_pos'].reshape(bs, n_kps, 3).clone().detach().cpu().numpy()[:8]
                
                gt_results, pred_results, true_blends_UV, pred_blends_UV = get_blended_images(gt_images, K, joints_3d_gt, joints_3d_pred, device)
                log_and_visualize_single(writer, global_iter, gt_results, pred_results,true_blends_UV, pred_blends_UV)
            
#            torch.cuda.synchronize()
#            t2 = time.time()
#            print("t3 - t1", t3 - t1)
#            print("t4 - t3", t4 - t3)
#            print("all_time", t2 - t1)
            
#            prev_time = t2
            
            global_iter += 1

                

        # Save Output and Checkpoint
        with torch.no_grad():
            # Save Checkpoint
            if dist.get_rank() == 0:
                print("save checkpoint")
                save_weights(os.path.join(checkpoint_path, "model.pth"), epoch, global_iter, model, voxel_optimizer, scheduler, cfg)
                if epoch % 2 == 0:
                    save_weights(os.path.join(checkpoint_path, "model_{}.pth".format(str(epoch).zfill(3))), epoch, global_iter, model, voxel_optimizer, scheduler, cfg) 
        
        
        #split = {"Validation" : [val_sampler, val_loader],}
        split = {"Real" : [real_sampler, real_loader], "Validation" : [val_sampler, val_loader]}
        #split = {"Real" : [real_sampler, real_loader],}
        heatmap_model.eval()
        #Validation
        if epoch % 2 == 0:
            with torch.no_grad():
                for mode, value in split.items():
                    sampler, loader = value[0], value[1]
                    sampler.set_epoch(epoch)
                    model.eval()
                    val_curr_loss = []
                    val_heatmap_loss = []
                    val_woff_loss = []
                    val_pos_loss = []
                    val_prob_loss = []
                    
                    val_acc = []
                    val_a1 = []
                    val_a2 = []
                    val_a3 = []
                    val_rmse = []
                    val_rmse_log = []
                    val_mae = []
                    val_abs_rel = []
                    val_add = []
                    
                    for batch_idx, batch in enumerate(tqdm(loader)):
    #                    if batch_idx <= 1124:
    #                        continue
                        curr_loss = 0.0
                    
                        # heatmap
    #                    input_tensor = batch['rgb_img'].to(device).float()
    #                    heatmap_gt = batch['heatmap_25d'].to(device).float()
    #                    b, c, h, w = heatmap_gt.size()
    #                    uv_ind = batch["uv_ind"].to(device).type(torch.int64)
    #                    uv_off = batch["uv_off"].to(device).float()
    #                    
    #                    
    #                    outputs = resnet_woff(input_tensor)
    #                    #outputs = model.module.forward_xyz(input_tensor)
    #                    torch.cuda.synchronize()
    #                    t2 = time.time()
    #                    heatmap_pred, off_pred, full_xyz_feat = outputs[0], outputs[1], outputs[2]
    #                    full_xyz_feat = F.interpolate(full_xyz_feat, (h, w), mode='bilinear', align_corners=False)
    #                    #print("full_xyz_feat.shape", full_xyz_feat.shape)
    #                    heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
    #                    heatmap_loss = loss_cfg["hm_coeff"] * heatmap_criterion(heatmap_gt, heatmap_pred)
    #                    
    #                    off_pred = F.interpolate(off_pred, (h, w), mode='bicubic',align_corners=False) # B x 2 x H x W
    #                    off_pred = off_pred.permute(0, 2, 3, 1).contiguous()
    #                    pred_uv_off = ((off_pred[:, :, :, :2]).view(b, -1, 2).contiguous())
    #                    uv_ind = uv_ind[:, :, :1].expand(uv_ind.size(0), uv_ind.size(1), pred_uv_off.size(2))
    #                    pred_uv_off = pred_uv_off.gather(1, uv_ind)
    #                    woff_loss = loss_cfg["woff_coeff"] * woff_criterion(pred_uv_off, uv_off)
    #                                
    #                    # prepare data
    #                    torch.cuda.synchronize()
    #                    t3 = time.time()
    
    
                        input_tensor = batch["xyz_img_scale"].float().to(device)
                        b, _, h, w = input_tensor.shape
                        heatmap_pred = heatmap_model(input_tensor)[-1]
                        #print("heatmap_pred.shape", heatmap_pred.shape)
                        heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.
                        #print("heatmap_pred.shape", heatmap_pred.shape)
                        _, c, _, _ = heatmap_pred.shape
                        
                        uv_pred = softargmax_uv(heatmap_pred[:, :c//2, :, :])
                        #print("uv_pred - uv_gt", uv_pred[0] - batch["joints_2D_uv"][0].to(device))
                        #print("uv gt", batch["joints_2D_uv"][0])
                        
                        batch["joints_2D_uv"] = uv_pred
#                        #print(batch["depth_path"])
#                        if "/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test/depth/000001.exr" in batch["depth_path"]:
#                            print("uv_pred", uv_pred[0])
#                            print("batch gt", batch["joints_2D_uv"][0])
                        
                        
                        data_dict = prepare_data(batch)
                        
                        # get valid pts other than kps
                        get_valid_points(data_dict, valid_sample_num=cfg["voxel_network"]["valid_sample_num"], rgb_in=cfg["voxel_network"]["rgb_in"])
                        
                        # get occupied voxel data
                        get_occ_vox_bound(data_dict, res=cfg["voxel_network"]["res"])
                        
                        # get kps ray data
                        get_miss_ray(data_dict)
                        
                        # ray AABB slab test
                        intersect_pair_flag = compute_ray_aabb(data_dict) 
                        
                        # compute gt
                        compute_gt(data_dict)
                        
                        # get embedding
                        #get_embedding_ours(data_dict, embed_fn, embeddirs_fn, full_xyz_feat, pnet_model)
                        get_embedding(data_dict, embed_fn, embeddirs_fn, resnet_model, pnet_model, local_embedding_type=cfg["voxel_network"]["local_embedding_type"]) 
                        
                        # get pred
                        get_pred(data_dict, "test", epoch-1, offset_dec, prob_dec, device, raw_input_type=cfg["voxel_network"]["raw_input_type"])
                        
                        torch.cuda.synchronize()
                        t4 = time.time()
                        # compute loss
                        loss_dict = compute_loss(data_dict, "test", epoch,device) 
                        
                        this_add = flat_add_from_pose(data_dict["pred_pos"].detach().cpu().numpy(), data_dict["gt_pos"].detach().cpu().numpy())
                        val_add = val_add + this_add
                        
                        
                                           
                        torch.distributed.barrier()
                        acc = reduce_mean(loss_dict["acc"], num_gpus)
                        a1 = reduce_mean(loss_dict["a1"], num_gpus)
                        a2 = reduce_mean(loss_dict["a2"], num_gpus)
                        a3 = reduce_mean(loss_dict["a3"], num_gpus)
                        rmse = reduce_mean(loss_dict["rmse"], num_gpus)
                        rmse_log = reduce_mean(loss_dict["rmse_log"], num_gpus)
                        mae = reduce_mean(loss_dict["mae"], num_gpus)
                        abs_rel = reduce_mean(loss_dict["abs_rel"], num_gpus)
                        
                        val_acc.append(acc.detach().cpu().numpy())
                        val_a1.append(a1.detach().cpu().numpy())
                        val_a2.append(a2.detach().cpu().numpy())
                        val_a3.append(a3.detach().cpu().numpy())
                        val_rmse.append(rmse.detach().cpu().numpy())
                        val_rmse_log.append(rmse_log.detach().cpu().numpy())
                        val_mae.append(mae.detach().cpu().numpy())
                        val_abs_rel.append(abs_rel.detach().cpu().numpy())
                        
                        
                        #print("loss_dict", loss_dict)
                        
                        pos_loss = loss_cfg["pos_coeff"] * loss_dict['pos_loss'] 
                        prob_loss = loss_cfg["prob_coeff"] * loss_dict['prob_loss']
                        
                        #curr_loss = heatmap_loss + woff_loss + pos_loss + prob_loss
                        curr_loss = pos_loss + prob_loss
    #                    
    #                    torch.cuda.synchronize()
    #                    t5 = time.time()
    #                    print("t5 - t4", t5 - t4)
    #                    print("t4 - t3", t4 - t3)
    #                    print("t3 - t2", t3 - t2)
    #                    print("t2 - t1", t2 - t1)
    #
    #
                        val_curr_loss.append(curr_loss.detach().item())
    #                    val_heatmap_loss.append(heatmap_loss.detach().item())
    #                    val_woff_loss.append(woff_loss.detach().item())
                        val_pos_loss.append(pos_loss.detach().item())
                        val_prob_loss.append(prob_loss.detach().item())
                    
                    torch.distributed.barrier()
                    val_add = distributed_concat(torch.from_numpy(np.array(val_add)).to(device), len(sampler.dataset))
                    
                    if dist.get_rank() == 0:
                        val_add = val_add.detach().cpu().numpy().tolist()
                        val_add_results = add_metrics(val_add, 0.06)
                        print(len(val_add))
                        print(val_add_results)
#                    
                        writer.add_scalar(f'{mode}/Validation Loss', np.mean(val_curr_loss), epoch)
    #                    writer.add_scalar(f'Validation/Validation Heatmap Loss', np.mean(val_heatmap_loss), epoch)
    #                    writer.add_scalar(f'Validation/Validation Woff Loss', np.mean(val_woff_loss), epoch)
                        writer.add_scalar(f'{mode}/Validation Pos Loss', np.mean(val_pos_loss), epoch)
                        writer.add_scalar(f'{mode}/Validation Prob Loss', np.mean(val_prob_loss), epoch)
                        
                        writer.add_scalar(f"{mode} Metrics/ ACC", np.mean(val_acc), epoch)
                        writer.add_scalar(f"{mode} Metrics/ a1", np.mean(val_a1), epoch)
                        writer.add_scalar(f"{mode} Metrics/ a2", np.mean(val_a2), epoch)
                        writer.add_scalar(f"{mode} Metrics/ a3", np.mean(val_a3), epoch)
                        writer.add_scalar(f"{mode} Metrics/ rmse", np.mean(val_rmse), epoch)
                        writer.add_scalar(f"{mode} Metrics/ rmse_log", np.mean(val_rmse_log), epoch)
                        writer.add_scalar(f"{mode} Metrics/ mae", np.mean(val_mae), epoch)
                        writer.add_scalar(f"{mode} Metrics/ abs_rel", np.mean(val_abs_rel), epoch)
                        
                        writer.add_scalar(f"{mode} Metrics/ ADD AUC", val_add_results["add_auc"], epoch)
                        writer.add_scalar(f"{mode} Metrics/ ADD MEDIAN", val_add_results["add_median"], epoch)
                        writer.add_scalar(f"{mode} Metrics/ ADD MEAN", val_add_results["add_mean"], epoch)
                        writer.add_scalar(f"{mode} Metrics/ ADD AUC THRESH", val_add_results["add_auc_thresh"], epoch)
                
        
        # Inference
#        if epoch % 1 == 0:
#            ass_add_results, ass_mAP_dict, angles_dict = network_inference(model, cfg, epoch, device)
#            if dist.get_rank() == 0:
#                visualize_inference_results(ass_add_results, ass_mAP_dict, angles_dict, writer, epoch)
        
#        scheduler.step(epoch + 1)

    
    



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfg, args = update_config()
    main(cfg)
#    find_seq_data_in_dir("/DATA/disk1/hyperplane/Depth_C2RP/Data/TY/")

















