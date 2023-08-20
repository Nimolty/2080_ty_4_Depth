#cuda_count=0,1,2,3,4,5
#num_samples_list="10"
#syn_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn/"
##model_list="model_00600.pth model_00700.pth model_00800.pth model_00900.pth model_01000.pth model_01100.pth model_01200.pth model_01300.pth model_01400.pth model_01500.pth model_01600.pth model_01700.pth model_01800.pth"
#model_list="model_01400.pth model_01500.pth model_01600.pth model_01700.pth model_01800.pth"
#for a in $model_list
#do
#  for syn_test in $syn_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=6 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net.py --exp_id diff_6 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_006_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --syn_test $syn_test --resume_checkpoint $a --num_samples $num_samples
#    done
#  done
#done 
cuda_count=0,1,2,3,4,5
num_samples_list="10"
syn_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test/depth/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/1_D415_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/2_D415_front_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/3_kinect_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/4_kinect_front_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/6_D415_left_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/8_kinect_left_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/9_kinect_left_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/10_kinect_right_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/12_D415_right_1/"
#model_list="model_00600.pth model_00700.pth model_00800.pth model_00900.pth model_01000.pth model_01100.pth model_01200.pth model_01300.pth model_01400.pth model_01500.pth model_01600.pth model_01700.pth model_01800.pth"
model_list="model_03350.pth"
for a in $model_list
do
  for syn_test in $syn_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=6 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_o6d.py --exp_id diff_7 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_007_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $syn_test --resume_checkpoint $a --num_samples $num_samples --pred_2d --eval_bs 30
    done
  done
done 
for a in $model_list
do
  for syn_test in $syn_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=6 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_o6d.py --exp_id diff_7 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_007_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $syn_test --resume_checkpoint $a --num_samples $num_samples 
    done
  done
done 
model_list="model_01100.pth"
for a in $model_list
do
  for syn_test in $syn_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=6 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net.py --exp_id diff_6 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_006_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $syn_test --resume_checkpoint $a --num_samples $num_samples --pred_2d --eval_bs 30
    done
  done
done 
for a in $model_list
do
  for syn_test in $syn_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=6 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net.py --exp_id diff_6 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_006_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results/results_3n_fixed_0425_right_RESNET_H_dz_075_/latest_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $syn_test --resume_checkpoint $a --num_samples $num_samples
    done
  done
done 