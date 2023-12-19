import warnings
warnings.filterwarnings('ignore', '.*FullyFusedMLP.*')

import torch
import time
import os
import numpy as np
from models.networks import NGP
from models.rendering import render
from metrics import psnr

from tqdm import tqdm
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from train import depth2img
import imageio
from datasets.opts import get_opts
import joblib


# dataset_name = 'nerf'
# scene = 'hotdog'
# dataset = dataset_dict[dataset_name](
#     f'/data/home/run/scv7303/rsw_/NeRFAttack/ngp_pl/dataset_source/{scene}',
#     split='test', downsample=1.0
# )

# exp_name = 'ngp_hotdog_nerf'
# model = NGP(scale=0.5).cuda()
# load_ckpt(model, f'ckpts/{dataset_name}/{exp_name}/epoch=19_slim.ckpt')

# psnrs = []; ts=0; imgs = []; depths = []
# os.makedirs(f'results/{dataset_name}/{scene}_traj', exist_ok=True)


# for img_idx in tqdm(range(len(dataset))):
#     t = time.time()
#     rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
#     results = render(model, rays_o, rays_d,
#                      **{'test_time': True,
#                         'T_threshold': 1e-2
#                         })
#     torch.cuda.synchronize()
#     ts += time.time()-t

#     pred = results['rgb'].reshape(dataset.img_wh[1], dataset.img_wh[0], 3).cpu().numpy()
#     pred = (pred*255).astype(np.uint8)
#     #depth = results['depth'].reshape(dataset.img_wh[1], dataset.img_wh[0]).cpu().numpy()
#     #depth_ = depth2img(depth)
#     imgs += [pred]
#     #depths += [depth_]
#     imageio.imwrite(f'results/{dataset_name}/{scene}_traj/{img_idx:03d}.png', pred)
#     #imageio.imwrite(f'results/{dataset_name}/{scene}_traj/{img_idx:03d}_d.png', depth_)

#     # if dataset.split != 'test_traj':
#     #     rgb_gt = dataset[img_idx]['rgb'].cuda()
#     #     psnrs += [psnr(results['rgb'], rgb_gt).item()]
# # if psnrs: print(f'mean PSNR: {np.mean(psnrs):.2f}, min: {np.min(psnrs):.2f}, max: {np.max(psnrs):.2f}')
# # print(f'mean time: {np.mean(ts):.4f} s, FPS: {1/np.mean(ts):.2f}')
# # print(f'mean samples per ray: {results["total_samples"]/len(rays_d):.2f}')

# if len(imgs)>30:
#     imageio.mimsave(f'results/{dataset_name}/{scene}_traj/rgb.mp4', imgs, fps=30)
#     # imageio.mimsave(f'results/{dataset_name}/{scene}_traj/depth.mp4', depths, fps=30)

# print('cost_time:', ts)
import time


@torch.no_grad()
def render_image(all_args, ckpt_path, is_over=False, is_viewfool=False):
    args = get_opts()

    dataset = dataset_dict[args.dataset_name](
        root_dir=args.root_dir,
        split='test', downsample=0.5, all_args=all_args, is_over=is_over, is_viewfool=is_viewfool
    )

    model = NGP(scale=0.5).cuda()
    load_ckpt(model, ckpt_path)

    imgs = []

    for img_idx in range(len(dataset)):
        rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
        results = render(model, rays_o, rays_d,
                        **{'test_time': True,
                            'T_threshold': 1e-2
                            })
        torch.cuda.synchronize()
        #TS += time.time()-t
        pred = results['rgb'].reshape(dataset.img_wh[1], dataset.img_wh[0], 3).cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        imgs += [pred]
    
    return imgs


