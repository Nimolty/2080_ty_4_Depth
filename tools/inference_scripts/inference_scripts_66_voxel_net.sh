#CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 20230 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_voxel_net.py --exp_id 66 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_66_voxel_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_refine --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth"

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_voxel_net.py --exp_id 66 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_66_voxel_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_refine --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth"

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_voxel_net.py --exp_id 63 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_63_voxel_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth"

#CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_voxel_net.py --exp_id 73 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_73_voxel_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume  --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth"

CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_voxel_net.py --exp_id 70 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_70_voxel_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume  --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" 