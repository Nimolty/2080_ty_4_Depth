#CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port 20232 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/train_spdh_toy.py --exp_id 09_01 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_50_spdh_resnet_h.yaml" --three_d_norm  --three_d_noise_std1 0.002 --three_d_noise_std2 0.002 --three_d_noise_std3 0.002 --three_d_random_drop 2.5e-5 --kps_14_name "09"

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port 20232 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/train_spdh_toy.py --exp_id 09_01 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_50_spdh_resnet_h.yaml" --three_d_norm  --three_d_noise_std1 0.002 --three_d_noise_std2 0.002 --three_d_noise_std3 0.002 --three_d_random_drop 2.5e-5 --kps_14_name "09" --three_d_noise_mu1 0.005 --three_d_noise_mu2 0.005 --three_d_noise_mu3 0.005

#CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port 20232 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/train_spdh_toy.py --exp_id 09_02 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_50_spdh_resnet_h.yaml" --three_d_norm  --three_d_noise_std1 0.002 --three_d_noise_std2 0.002 --three_d_noise_std3 0.002 --three_d_random_drop 2.5e-5 --kps_14_name "09"