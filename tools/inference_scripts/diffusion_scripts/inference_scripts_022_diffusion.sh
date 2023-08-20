cuda_count=4,5,6,7
num_samples_list="10"
model_list="model_00400.pth"
real_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test/depth/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/1_D415_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/2_D415_front_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/3_kinect_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/4_kinect_front_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/6_D415_left_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/8_kinect_left_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/9_kinect_left_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/10_kinect_right_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/12_D415_right_1/"
for a in $model_list
do
  for real_test in $real_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=4 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_o6d.py --exp_id diff_22 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_022_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples 
    done
  done
done
for a in $model_list
do
  for real_test in $real_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=4 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_o6d.py --exp_id diff_22 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_022_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --eval_bs 30 --pred_2d 
    done
  done
done
for a in $model_list
do
  for real_test in $real_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=4 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_o6d.py --exp_id diff_22 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_022_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --eval_bs 30 --pred_2d --change_intrinsic
    done
  done
done        
