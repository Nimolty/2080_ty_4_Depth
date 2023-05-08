# exp_id 52 : res : 12, global feat + local feat global_ratio 64 sigmoid=True
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 --master_port 20202 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/train_voxel_net.py --exp_id 52 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_51_voxel_net.yaml" --resume 
