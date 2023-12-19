#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2020.11
source activate fastNeRF


# 1
python train_trades_imagenet_viewpoint_new.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --AT_exp_name 'final_res_viatvf_R' --lr 0.001 --epochs 90 --num_k 15 --no_background --fast_AVDT --share_dist --AT_type 'AVDT' --share_dist_rate 0.5 --ckpt_attack_path './run_train_NeRF/ckpts/nerf/dataset_all' --treat_model 'resnet50'
