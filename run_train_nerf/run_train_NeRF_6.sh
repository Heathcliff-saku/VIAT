#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2020.11
module load gcc/9.3
module load CUDA/11.3.1
source activate fastNeRF

# export CUDA_HOME=/data/apps/CUDA/11.3.1/
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
# export PATH=${CUDA_HOME}/bin:${PATH}
# export PATH
# export LD_LIBRARY_PATH=/data/apps/gcc/9.3.0/lib64:$LD_LIBRARY_PATH    


# python train.py --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/Synthetic_NeRF/Hotdog' --exp_name 'ngp_hotdog' --num_epochs 20 --batch_size 16384 --lr 2e-2



python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_01' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_02' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_03' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_04' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_05' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_06' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_07' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_08' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_09' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2

python train.py --dataset_name nerf --root_dir '/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/GMFool_dataset/sign_10' --exp_name 'new/88' --num_epochs 5 --batch_size 16384 --lr 2e-2