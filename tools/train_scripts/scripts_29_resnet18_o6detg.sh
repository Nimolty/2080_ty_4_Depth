#CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 train_o6d.py --exp_id 29 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_29_resnet18_o6detg.yaml" --resume
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 train_o6d.py --exp_id 29 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_29_resnet18_o6detg.yaml" 