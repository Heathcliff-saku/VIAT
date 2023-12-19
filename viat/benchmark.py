from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from trades import trades_loss

from PIL import Image  
import numpy as np
import torchvision.models as models
from models.networks import NGP
from models.rendering import render
from metrics import psnr

from tqdm import tqdm
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from train import depth2img
import imageio
import joblib
import timm

from robustness import model_utils
from robustness.datasets import ImageNet


from datasets.opts import get_opts

# 用于寻找攻击的 dist_pool 需要遍历当前所有存在的物体 保存dist_pool.npy（为了warm-star操作）
from NES_GMM_forAT import NES_GMM_search
from NES_viewfool_forAT import NES_viewfool_search


args = get_opts()
def get_black_model(model_name):
    checkpoint = None
    flag = 0
    # vgg
    if model_name == 'vgg16':
        model = timm.create_model('vgg16', pretrained=True)
    if model_name == 'vgg19':
        model = timm.create_model('vgg19', pretrained=True)

    # resnet:
    if model_name == 'resnet18':
        model = timm.create_model('resnet18', pretrained=True)
    if model_name == 'resnet34':
        model = timm.create_model('resnet34', pretrained=True)
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
    if model_name == 'resnet101':
        model = timm.create_model('resnet101', pretrained=True)
    if model_name == 'resnet152':
        model = timm.create_model('resnet152', pretrained=True)
    
    # inc
    if model_name == 'inc_v3':
        model = timm.create_model('inception_v3', pretrained=True)
    if model_name == 'inc_v4':
        model = timm.create_model('inception_v4', pretrained=True)
    if model_name == 'inc_res_v2':
        model = timm.create_model('inception_resnet_v2', pretrained=True)

    # dense
    if model_name == 'densenet121':
        model = timm.create_model('densenet121', pretrained=True)
    if model_name == 'densenet169':
        model = timm.create_model('densenet169', pretrained=True)
    if model_name == 'densenet201':
        model = timm.create_model('densenet201', pretrained=True)

    # effe
    if model_name == 'en_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True)
    if model_name == 'en_b1':
        model = timm.create_model('efficientnet_b1', pretrained=True)
    if model_name == 'en_b2':
        model = timm.create_model('efficientnet_b2', pretrained=True)
    if model_name == 'en_b3':
        model = timm.create_model('efficientnet_b3', pretrained=True)
    if model_name == 'en_b4':
        model = timm.create_model('efficientnet_b4', pretrained=True)

    # mobile
    if model_name == 'mobilenetv2_120d':
        model = timm.create_model('mobilenetv2_120d', pretrained=True)
    if model_name == 'mobilenetv2_140':
        model = timm.create_model('mobilenetv2_140', pretrained=True)
    
    # vit
    if model_name == 'vit_base':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    if model_name == 'vit_large':
        model = timm.create_model('vit_large_patch16_224', pretrained=True)

    # deit
    if model_name == 'deit_tiny':
        model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    if model_name == 'deit_small':
        model = timm.create_model('deit_small_patch16_224', pretrained=True)
    if model_name == 'deit_base':
        model = timm.create_model('deit_base_patch16_224', pretrained=True)

    # swin
    if model_name == 'swin_tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    if model_name == 'swin_small':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    if model_name == 'swin_base':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    if model_name == 'swin_large':
        model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)

    # mixer
    if model_name == 'mixer_b16':
        model = timm.create_model('mixer_b16_224', pretrained=True)
    if model_name == 'mixer_l16':
        model = timm.create_model('mixer_l16_224', pretrained=True)
    
    # augmix
    if model_name == 'augmix':
        model = timm.create_model('resnet50', pretrained=False)
        checkpoint = '/data/home/scv7303/.cache/torch/hub/checkpoints/augmix.pth.tar'
        flag = 1
    if model_name == 'deepaugment':
        model = timm.create_model('resnet50', pretrained=False)
        checkpoint = '/data/home/scv7303/.cache/torch/hub/checkpoints/deepaugment.pth.tar'
        flag = 1
    if model_name == 'augmix+deepaugment':
        model = timm.create_model('resnet50', pretrained=False)
        checkpoint = '/data/home/scv7303/.cache/torch/hub/checkpoints/deepaugment_and_augmix.pth.tar'
        flag = 1

    # l2robust
    if model_name == 'resnet50_l2_robust_eps=1.0':
        model = timm.create_model('resnet50', pretrained=False)
        checkpoint = '/data/home/scv7303/.cache/torch/hub/checkpoints/resnet50_l2_eps1.ckpt'
        flag = 2
    if model_name == 'resnet50_l2_robust_eps=3.0':
        model = timm.create_model('resnet50', pretrained=False)
        checkpoint = '/data/home/scv7303/.cache/torch/hub/checkpoints/resnet50_l2_eps3.ckpt'
        flag = 2
    if model_name == 'resnet50_l2_robust_eps=5.0':
        model = timm.create_model('resnet50', pretrained=False)
        checkpoint = '/data/home/scv7303/.cache/torch/hub/checkpoints/resnet50_l2_eps5.ckpt'
        flag = 2

    # VIAT
    if model_name == 'resnet50-standard':
        model = timm.create_model('resnet50', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-res50-epoch120.pt'
        model.load_state_dict(torch.load(checkpoint))

    if model_name == 'resnet50-VIAT-gf':
        model = timm.create_model('resnet50', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-resnet50-final_res_viatgf_R-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))

    if model_name == 'resnet50-VIAT-vf':
        model = timm.create_model('resnet50', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-resnet50-final_res_viatvf_R-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))

    if model_name == 'resnet50-VIAT-natural':
        model = timm.create_model('resnet50', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-resnet50-final_res_natural-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))

    if model_name == 'resnet50-VIAT-random':
        model = timm.create_model('resnet50', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-resnet50-final_res_random-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))

    if model_name == 'vit-standard':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-vit-b-vit-b_pre_train-epoch10.pt'
        model.load_state_dict(torch.load(checkpoint))
    
    if model_name == 'vit-VIAT-gf':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-vit-b-final_vit_viatgf_R-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))
    
    if model_name == 'vit-VIAT-vf':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-vit-b-final_vit_viatvf_R-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))
    
    if model_name == 'vit-VIAT-natural':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-vit-b-final_vit_natural-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))
    
    if model_name == 'vit-VIAT-random':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-vit-b-final_vit_random-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))

    if model_name == 'inc-VIAT-gf':
        # model = timm.create_model('inception_v3', pretrained=True, num_classes=100)
        model = torchvision.models.inception_v3(pretrained=False)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-inc-v3-final_inc_viatgf_R-epoch60.pt'
        #----------------------------------------------
        num_ftr = model.fc.in_features
        model.fc = nn.Linear(num_ftr,100)
        model.load_state_dict(torch.load(checkpoint))

    if model_name == 'inc-res-VIAT-gf':
        model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-inc-res-inc-res_viatgf_R-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))
    if model_name == 'en-VIAT-gf':
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-en-en_viatgf_R-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))
    if model_name == 'dn-VIAT-gf':
        model = timm.create_model('densenet121', pretrained=False, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-dn-dn_viatgf_R-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))
    if model_name == 'deit-VIAT-gf':
        model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-deit-deit_viatgf_R-epoch60.pt'
        model.load_state_dict(torch.load(checkpoint))
    if model_name == 'swin-VIAT-gf':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=100)
        checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-swin-swin_viatgf_R-epoch40.pt'
        model.load_state_dict(torch.load(checkpoint))
        

    if checkpoint != None and flag==1:
        check = torch.load(checkpoint)
        arch = check['arch']
        model = models.__dict__[arch]()
        model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0]).cuda()
        model.load_state_dict(check['state_dict'])
        model = model.to(device)
    
    # l2 robust
    elif checkpoint != None and flag==2:
        imagenet_ds = ImageNet('data/pathl')
        model , _ = model_utils.make_and_restore_model(arch="resnet50", dataset=imagenet_ds, 
resume_path=checkpoint, parallel=False, add_custom_forward=True)
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        # model.eval()
        # model.cuda()
        # transform = transforms.Compose([transforms.Resize((248, 248)),transforms.CenterCrop(224),transforms.ToTensor(), ])

    else:
    # model.load_state_dict(torch.load(checkpoint))
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    
    model.eval()

    return model


# ------------------------------------------------------------------------------------------------------------- #


# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device=torch.device("cuda:0" )
gpus = [0, 1]
# gpus = [0, 1, 2, 3]

kwargs = {'num_workers': 10*len(gpus), 'pin_memory': True} if use_cuda else {}

# setup data loader

valdir = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/adv_view_all/adv_view_gmfool_vit'

test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                                      std=[0.229, 0.224, 0.225])
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)





def eval_test(model_name, map_label, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # print(target)
            # for i in range(len(target)):
            #     map_label_ = map_label[target[i].numpy()]
            #     target[i] = int(map_label_)
            # print(target.shape)
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output.shape)
            # new_output = torch.zeros((num,100), dtype=torch.float).to(device)
            # for k in range(num):
            #     for i in range(len(map_label)):
            #         for j in range(len(output)):
            #             if j == map_label[i]:
            #                 new_output[k,i] = output[k, j]
            # print(output)
            # print(output[0,403])
            # print(new_output)
            # print(new_output[0, 1])
            # output = new_output
            if type(output) == tuple:
                output = output[0]
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('result to {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        model_name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy




def main():
    with open('/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/label_map.txt') as f:
        map_label = [line.strip() for line in f.readlines()]

    # model_list = ['augmix','deepaugment','augmix+deepaugment','resnet50_l2_robust_eps=1.0','resnet50_l2_robust_eps=3.0','resnet50_l2_robust_eps=5.0','vgg16','vgg19','resnet18','resnet34','resnet50','resnet101','resnet152','inc_v3','inc_v4','inc_res_v2','densenet121','densenet169','densenet201','en_b0','en_b1','en_b2','en_b3','en_b4','mobilenetv2_120d','mobilenetv2_140','vit_base','vit_large','deit_tiny','deit_small','deit_base','swin_tiny','swin_small','swin_base','swin_large','mixer_b16','mixer_l16']
    
    # model_list = ['resnet50', 'inc_v3', 'vit_base', 'inc_res_v2', 'densenet121', 'en_b0', 'deit_base', 'swin_base']
    # model_list = ['resnet50-VIAT', 'vit-VIAT']

    model_list = ['deit-VIAT-gf', 'swin-VIAT-gf']
    
    for i in tqdm(range(len(model_list))):
        model = get_black_model(model_list[i])
        # evaluation on natural examples
        print('================================================================')
        eval_test(model_list[i], map_label, model, device, test_loader)
        print('================================================================')


if __name__ == '__main__':
    main()
