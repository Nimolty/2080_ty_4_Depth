CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 --master_port 20230 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/train_voxel_net.py --exp_id 80 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_80_voxel_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" 