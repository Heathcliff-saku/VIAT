#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2020.11
source activate fastNeRF

python Attack_exp_fast_K.py --root_dir '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog' --dataset_name nerf_for_attack --scene_name 'resnet_GMM/hotdog' --N_importance 64 --ckpt_path '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/ckpts/nerf/ngp_hotdog_nerf/epoch=19_slim.ckpt' --optim_method NES --search_num 6 --popsize 101 --iteration 50 --iteration_warmstart 10 --mu_lamba 0.05 --sigma_lamba 0.05 --omiga_lamba 0.05 --num_sample 100 --train_mood 'AT' --batch-size 512 --test-batch-size 512 --lr 0.001 --epochs 90 --no_background --share_dist --ckpt_attack_path './run_train_NeRF/ckpts/nerf/dataset_all' --treat_model 'resnet50' --AT_exp_name 'k5' --num_k 5

