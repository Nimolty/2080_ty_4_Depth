cuda_count=1
num_samples_list="1"
model_list="model_02000.pth"
#real_tests="/DATA/disk1/hyperplane/ty_data/panda-3cam_realsense/ /DATA/disk1/hyperplane/ty_data/panda-3cam_azure/ /DATA/disk1/hyperplane/ty_data/panda-orb/"
real_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/1_D415_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/2_D415_front_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/3_kinect_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/4_kinect_front_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/6_D415_left_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/8_kinect_left_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/9_kinect_left_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/10_kinect_right_1/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/12_D415_right_1/"
#real_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/3_kinect_front_0/ /DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/4_kinect_front_1/"

for a in $model_list
do
  for real_test in $real_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_rtm_single.py --exp_id diff_37 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_rtm_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_032_tra/epoch_017_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples --eval_bs 30  --load_mask  --rgb --pred_2d --pred_mask --num_samples $num_samples --gt_angle --sampler ode
    done
  done
done
