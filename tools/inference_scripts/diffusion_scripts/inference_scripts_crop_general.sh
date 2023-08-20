cuda_count=0,7
num_samples_list="10"
model_list="model_01000.pth"
##real_tests="/DATA/disk1/hyperplane/ty_data/panda-3cam_realsense/ /DATA/disk1/hyperplane/ty_data/panda-3cam_kinect360/ /DATA/disk1/hyperplane/ty_data/panda-3cam_azure/ /DATA/disk1/hyperplane/ty_data/panda-orb/"
real_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/1_D415_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/2_D415_front_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/3_kinect_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/4_kinect_front_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/6_D415_left_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/8_kinect_left_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/9_kinect_left_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/10_kinect_right_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/12_D415_right_1/"
#model_list="model_00100.pth model_00200.pth model_00300.pth model_00400.pth model_00500.pth model_00600.pth model_00700.pth model_00800.pth model_00900.pth model_01000.pth model_01100.pth model_01200.pth model_01300.pth model_01400.pth model_01500.pth model_01600.pth model_01700.pth model_01800.pth model_01900.pth model_02000.pth model_02100.pth model_02200.pth model_02300.pth model_02400.pth"
#model_list="model_00900.pth model_01100.pth model_01200.pth model_01600.pth model_01900.pth model_02000.pth model_02400.pth"
#real_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/1_D415_front_0/"
#real_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/1_D415_front_0/"
#
##model_list="model_00125.pth" 
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=4 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_single.py --exp_id diff_26 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples  --sampler ode
#    done
#  done
#done
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=5 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_single.py --exp_id diff_26 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples  --sampler pc 
#    done
#  done
#done
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=7 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_crop.py --exp_id diff_26 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --sampler ode --eval_bs 30 
#    done
#  done
#done
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=7 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_single.py --exp_id diff_37 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --sampler ddim_solver 
#    done
#  done
#done
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=2 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_crop_general.py --exp_id general_02 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_general_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --eval_bs 30  --load_mask  --rgb --pred_2d --pred_mask --infer_prob 0.0
#    done
#  done
#done
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=2 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_crop_general.py --exp_id general_02 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_general_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --eval_bs 30  --load_mask  --rgb --pred_2d --pred_mask --infer_prob 1.0
#    done
#  done
#done
for a in $model_list
do
  for real_test in $real_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=2 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_crop_general.py --exp_id general_05 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_general_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --eval_bs 30  --load_mask  --rgb   
    done
  done
done
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=7 --master_port 20231 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_crop.py --exp_id diff_37 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --eval_bs 30  --load_mask  --rgb --pred_2d --pred_mask
#    done
#  done
#done
