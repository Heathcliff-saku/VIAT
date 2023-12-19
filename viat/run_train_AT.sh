#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2020.11
source activate fastNeRF

# python train_trades_imagenet_viewpoint_new.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --num_k 1 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --lr 0.001 --epochs 90 --AT_exp_name 'test' --no_background --fast_AVDT --share_dist --AT_type 'AVDT' --share_dist_rate 0.5 --ckpt_attack_path './ckpts/nerf'

#-------------------------------

# 1
python train_trades_imagenet_viewpoint_new.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --AT_exp_name 'test_k=1' --lr 0.001 --epochs 1 --num_k 1 --no_background --fast_AVDT --share_dist --AT_type 'AVDT' --share_dist_rate 0 --ckpt_attack_path './run_train_NeRF/ckpts/nerf/dataset_all'

# 2
#python train_trades_imagenet_viewpoint_new.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --AT_exp_name 'test_k=5' --lr 0.001 --epochs 90 --num_k 5 --no_background --fast_AVDT --share_dist --AT_type 'AVDT' --share_dist_rate 0 --ckpt_attack_path './run_train_NeRF/ckpts/nerf/dataset_all'

# 3
#python train_trades_imagenet_viewpoint_new.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --AT_exp_name 'test_k=10' --lr 0.001 --epochs 90 --num_k 10 --no_background --fast_AVDT --share_dist --AT_type 'AVDT' --share_dist_rate 0 --ckpt_attack_path './run_train_NeRF/ckpts/nerf/dataset_all'

# 4
#python train_trades_imagenet_viewpoint_new.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --AT_exp_name 'test_k=15' --lr 0.001 --epochs 90 --num_k 15 --no_background --fast_AVDT --share_dist --AT_type 'AVDT' --share_dist_rate 0 --ckpt_attack_path './run_train_NeRF/ckpts/nerf/dataset_all'

# 5
#python train_trades_imagenet_viewpoint_new.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --AT_exp_name 'test_k=20' --lr 0.001 --epochs 90 --num_k 20 --no_background --fast_AVDT --share_dist --AT_type 'AVDT' --share_dist_rate 0 --ckpt_attack_path './run_train_NeRF/ckpts/nerf/dataset_all'

# 6
#python train_trades_imagenet_viewpoint_new.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --AT_exp_name 'test_k=25' --lr 0.001 --epochs 90 --num_k 25 --no_background --fast_AVDT --share_dist --AT_type 'AVDT' --share_dist_rate 0 --ckpt_attack_path './run_train_NeRF/ckpts/nerf/dataset_all'

#----------------------------------------------------------------
# --no_background  --share_dist   './run_train_NeRF/ckpts/nerf/dataset_all'

# 第一次实验全部是 lr=0.001  376570 376370   
# 第二次开始设置 lr=0.01  377804
# lr=0.1 377805