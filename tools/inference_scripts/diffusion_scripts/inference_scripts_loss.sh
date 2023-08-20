cuda_count=1,2,3,4,5,6,7
num_samples_list="10"
#model_list="model_00100.pth model_00200.pth model_00300.pth model_00400.pth model_00500.pth model_00600.pth model_00700.pth model_00800.pth model_00900.pth model_01000.pth model_01100.pth model_01200.pth model_01300.pth model_01400.pth model_01500.pth model_01600.pth model_01700.pth model_01800.pth model_01900.pth model_02000.pth model_02100.pth model_02200.pth model_02300.pth model_02400.pth"
model_list="model_01900.pth model_02000.pth model_02100.pth model_02200.pth model_02300.pth model_02400.pth"
real_tests="/DATA/disk1/hyperplane/Depth_C2RP/Data/Real_Test_0613/1_D415_front_0/"
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=7 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_loss.py --exp_id diff_26 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples  --sampler ode --training_loss_start_idx 0.00001 --training_loss_end_idx 0.001
#    done
#  done
#done
#for a in $model_list
#do
#  for real_test in $real_tests
#  do
#    for num_samples in $num_samples_list
#    do
#      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=7 --master_port 22203 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_loss.py --exp_id diff_26 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_crop_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples  --sampler ode --training_loss_start_idx 0.001 --training_loss_end_idx 1.0
#    done
#  done
#done
for a in $model_list
do
  for real_test in $real_tests
  do
    for num_samples in $num_samples_list
    do
      CUDA_VISIBLE_DEVICES=$cuda_count python -m torch.distributed.launch --nproc_per_node=7 --master_port 22322 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_diffusion_net_loss.py --exp_id diff_39 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_scripts/diffusion_scripts/inference_037_diff_net.yaml"  --resume_heatmap "/DATA/disk1/hyperplane/Depth_C2RP/Code/Baselines/spdh/rpe_spdh/results_crop_2Ds/results_007_tra/epoch_020_checkpoint.pth" --resume --resume_simplenet "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth" --real_test $real_test --resume_checkpoint $a --num_samples $num_samples  --sampler ode --training_loss_start_idx 0.001 --training_loss_end_idx 1.0
    done
  done
done
