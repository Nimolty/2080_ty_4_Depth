CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 train_o6d_reset.py --exp_id 34 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_34_resnet18_o6ddisetg.yaml" --resume