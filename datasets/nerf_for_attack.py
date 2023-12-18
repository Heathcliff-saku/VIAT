import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image
from .opts import get_opts
from .base import BaseDataset

def pose_spherical(all_args):
    args = get_opts()
    search_num = args.search_num

    if search_num == 3:
        theta, phi, radius = all_args
    if search_num == 6 or search_num == 123 or search_num == 456 or search_num == 1231:
        gamma, theta, phi, radius, x, y = all_args

    trans_t = lambda t, x, y: torch.Tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, t],
        [0, 0, 0, 1]]).float()

    rot_phi = lambda phi: torch.Tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]]).float()

    rot_gamma = lambda gamma: torch.Tensor([
        [np.cos(gamma), 0, -np.sin(gamma), 0],
        [0, 1, 0, 0],
        [np.sin(gamma), 0, np.cos(gamma), 0],
        [0, 0, 0, 1]]).float()

    rot_theta = lambda th: torch.Tensor([
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]).float()

    if search_num == 3:
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w

    if search_num == 6 or search_num == 123 or search_num == 456 or search_num == 1231:
        c2w = trans_t(radius, x, y)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_gamma(gamma / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w

    return c2w


class NeRFAttackDataset(BaseDataset):
    def __init__(self, all_args, root_dir, split='train', downsample=1.0, is_over=False, is_viewfool=False, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.is_over = is_over
        self.all_args = all_args
        self.read_intrinsics()
        self.split = split
        self.is_viewfool = is_viewfool
        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
            meta = json.load(f)

        w = h = int(800*self.downsample)
        fx = fy = 0.5*800/np.tan(0.5*meta['camera_angle_x'])*self.downsample

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def __len__(self):
        args = get_opts()
        if args.optim_method == 'random':
            return args.num_sample
        if args.optim_method == 'train_pose':
            return 1
        else:
            if self.is_over:
                a = args.num_sample
            else:
                a = args.popsize-1+args.num_k
                if self.split == 'AT' or self.is_viewfool:
                    return len(self.all_args)
                else:
                    return a
        

    def read_meta(self, split):
        self.rays = []
        self.poses = []
        if split == 'AT':
            random = self.all_args  # 若搜素结束，则生成random的100张图像
            render_poses_ = torch.stack([pose_spherical([gamma_, th_, phi_, r_, a_, b_]) for gamma_, th_, phi_, r_, a_, b_ in random], 0)

        if split == 'test':
            if self.is_over:
                random = self.all_args  # 若搜素结束，则生成random的100张图像
                render_poses_ = torch.stack([pose_spherical([gamma_, th_, phi_, r_, a_, b_]) for gamma_, th_, phi_, r_, a_, b_ in random], 0)

            else:
                random = self.all_args  # 若搜素结束，则生成random的100张图像
                render_poses_ = torch.stack([pose_spherical([gamma_, th_, phi_, r_, a_, b_]) for gamma_, th_, phi_, r_, a_, b_ in random], 0) # 若是正常的迭代过程中，只需生成对应角度的一张图像


        if split == 'test' or split == 'AT':
            pose_radius_scale = 1.5
            for frame in range(len(render_poses_)):
                render_poses_[frame, :, 1:3] *= -1 # [right up back] to [right down front]
                render_poses_[frame, :, 3] /= np.linalg.norm(render_poses_[frame, :, 3])/pose_radius_scale
            self.poses = render_poses_[:, :3, :]
        

        else:
            for frame in tqdm(frames):
                c2w = np.array(frame['transform_matrix'])[:3, :4]

                # determine scale
                if 'Jrender_Dataset' in self.root_dir:
                    c2w[:, :2] *= -1 # [left up front] to [right down front]
                    folder = self.root_dir.split('/')
                    scene = folder[-1] if folder[-1] != '' else folder[-2]
                    if scene=='Easyship':
                        pose_radius_scale = 1.2
                    elif scene=='Scar':
                        pose_radius_scale = 1.8
                    elif scene=='Coffee':
                        pose_radius_scale = 2.5
                    elif scene=='Car':
                        pose_radius_scale = 0.8
                    else:
                        pose_radius_scale = 1.5
                else:
                    c2w[:, 1:3] *= -1 # [right up back] to [right down front]
                    pose_radius_scale = 1.5
                c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/pose_radius_scale

                # add shift
                if 'Jrender_Dataset' in self.root_dir:
                    if scene=='Coffee':
                        c2w[1, 3] -= 0.4465
                    elif scene=='Car':
                        c2w[0, 3] -= 0.7
                self.poses += [c2w]

                try:
                    img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                    img = read_image(img_path, self.img_wh)
                    self.rays += [img]
                except: pass

        # if len(self.rays)>0:
        #     self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        # # self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)

