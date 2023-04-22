#CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh.py --exp_id 37 --epoch_id 260 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn" --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/38/CHECKPOINT/model_003.pth"
#CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_multi.py --exp_id 38 --epoch_id 5 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn" --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/38/CHECKPOINT/model.pth" --dr_iter_num 3
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_multi.py --exp_id 39 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn" --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/39/CHECKPOINT/model.pth" --dr_iter_num 0 --link_idx 1
#
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_multi.py --exp_id 39 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn" --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/39/CHECKPOINT/model.pth" --dr_iter_num 1 --link_idx 1

#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 30302 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_multi.py --exp_id 39 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/39/CHECKPOINT/model.pth" 

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 30302 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_multi.py --exp_id 40 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/40/CHECKPOINT/model.pth" --dr_iter_num 2 --link_idx 0


#CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_single.py --exp_id 40 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/40/CHECKPOINT/model.pth" 

#
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_multi.py --exp_id 39 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn" --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/39/CHECKPOINT/model.pth" --dr_iter_num 5 --link_idx 6

#CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_curr.py --exp_id 39 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml"  --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn" --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/39/CHECKPOINT/model.pth" --load_current_predgt "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/39/INFERENCE_LOGS/EXP39_001_0.json"

#CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_toy.py --exp_id 49 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_37_spdh_resnet_h.yaml" --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/49/CHECKPOINT/model.pth"  --three_d_norm


#CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_toy.py --exp_id 50 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_50_spdh_resnet_h.yaml" --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/50/CHECKPOINT/model.pth"  --three_d_norm --three_d_noise_mu 0.005 --three_d_noise_std 0.002 --three_d_random_drop 2.5e-5
##
##
#CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_toy.py --exp_id 51 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_50_spdh_resnet_h.yaml" --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/51/CHECKPOINT/model.pth"  --three_d_norm --three_d_noise_mu 0.005 --three_d_noise_std 0.002 --three_d_random_drop 2.5e-5

CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_toy.py --exp_id 52 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_50_spdh_resnet_h.yaml" --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/52/CHECKPOINT/model.pth"  --three_d_norm --three_d_noise_mu1 0.005 --three_d_noise_mu2 0.008 --three_d_noise_mu3 0.03 --three_d_noise_std1 0.005 --three_d_noise_std2 0.015 --three_d_noise_std3 0.04 --three_d_random_drop 2.5e-5 

#CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_toy.py --exp_id 53 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_50_spdh_resnet_h.yaml" --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/53/CHECKPOINT/model.pth"  --three_d_norm --three_d_noise_mu 0.008 --three_d_noise_std 0.002 --three_d_random_drop 2.5e-5

CUDA_VISIBLE_DEVICES=0 python /DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/tools/inference_spdh_toy.py --exp_id 54 --epoch_id 1 --cfg "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/configs/train_50_spdh_resnet_h.yaml" --syn_test "/DATA/disk1/hyperplane/Depth_C2RP/Data/Data_0201_Syn"  --model_path "/DATA/disk1/hyperplane/Depth_C2RP/Code/Ours_Code/output/toy/54/CHECKPOINT/model.pth"  --three_d_norm --three_d_noise_mu1 0.005 --three_d_noise_mu2 0.008 --three_d_noise_mu3 0.03 --three_d_noise_std1 0.005 --three_d_noise_std2 0.015 --three_d_noise_std3 0.04 --three_d_random_drop 2.5e-5 









