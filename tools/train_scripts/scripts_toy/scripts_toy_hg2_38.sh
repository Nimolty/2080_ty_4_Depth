CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 20320 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/train_spdh_toy.py --exp_id 38 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_38_spdh_hg2.yaml" --resume 