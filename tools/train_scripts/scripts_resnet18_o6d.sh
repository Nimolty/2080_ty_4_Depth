CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 train_o6d.py --exp_id 26 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_resnet18_o6d.yaml" --resume  