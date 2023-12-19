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

from datasets.opts import get_opts

# 用于寻找攻击的 dist_pool 需要遍历当前所有存在的物体 保存dist_pool.npy（为了warm-star操作）
from NES_GMM_forAT import NES_GMM_search
from NES_viewfool_forAT import NES_viewfool_search


args = get_opts()

@torch.no_grad()
def render_image(all_args, labels, objects, is_over=False, split='train'):
    """
    调用渲染函数 从viewpoints渲染一批图像
    Args:
        all_args: viewpoints array(batchsize, 6)
        labels: 每列viewpoint属于的类别号 array(batchszie, 1)
        batch_size: 每列viewpoint属于的类别中的物体序号 array(batchszie, 1)

    Returns: a batch of adversarial viewpoint rendering images

    ./ckpt/nerf/ 存放着所有物体的nerf权重
    -----------
    /913/00.pt
    /913/01.pt
    ... ... ...
    /913/09.pt
    -------------
    """

    dataset = dataset_dict['nerf_for_attack'](
        root_dir='/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/dataset_source/viewfool/hotdog',
        split='AT', downsample=0.5, all_args=all_args, is_over=is_over
    )
    model = NGP(scale=0.5).cuda()
    
    # save_path = f'results/{args.dataset_name}/{args.scene_name}'
    # os.makedirs(save_path, exist_ok=True)
    imgs = np.zeros((len(dataset), 400, 400, 3))
    
    for img_idx in range(len(dataset)):
        
        ckpt_path = f'./ckpts/nerf/{split}/' + str('%02d' % int(labels[img_idx])) + '/' + str('%02d' % objects[img_idx]) + '.ckpt'
        load_ckpt(model, ckpt_path)

        rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
        results = render(model, rays_o, rays_d,
                        **{'test_time': True,
                            'T_threshold': 1e-2
                            })
        torch.cuda.synchronize()
        #TS += time.time()-t
        pred = results['rgb'].reshape(dataset.img_wh[1], dataset.img_wh[0], 3).cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        imgs[img_idx, :, :, :] = pred
        # if is_over:
        #     imageio.imwrite(os.path.join(save_path, f'{img_idx:03d}.png'), pred)

    return imgs


def AddBackground(render_imgs, clean_images, batch_size, split, flag='AT'):
    
    render_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.crop_size),
            transforms.ToTensor(),
        ])

    render_imgs_ = torch.zeros([len(render_imgs), 3, args.crop_size, args.crop_size])
    for i in range(len(render_imgs)):
        a = np.uint8(render_imgs[i,:,:,:])
        render_imgs_[i,:,:,:] = render_transform(a)
        #--------------------------------------------------------------------#
        # save = render_imgs_[i,:,:,:].numpy()
        # save = np.transpose(save, (1,2,0))
        # save = (save*255).astype(np.uint8)
        # # print(save)
        # # print(save.shape)
        # imageio.imwrite(os.path.join('./results', f'{i:03d}.png'), save)
        #--------------------------------------------------------------------#
        
    if split == 'train' and not args.no_background and flag == 'AT':
        for i in range(batch_size):
            background = torch.squeeze(clean_images[np.random.randint(low=0, high=batch_size, size=1),:,:,:])
            render = render_imgs_[i, :, :, :]
            # print('background',background.size())
            # print('render', render_imgs_[i, :, :, :].size())
            
            for h in range(args.crop_size):
                for w in range(args.crop_size):
                    if render[0, h, w]>0.95:
                        render_imgs_[i, :, h, w] = background[:, h, w]
        #--------------------------------------------------------------------#
    save = render_imgs_[i,:,:,:].numpy()
    save = np.transpose(save, (1,2,0))
    save = (save*255).astype(np.uint8)
    # print(save)
    # print(save.shape)
    imageio.imwrite(os.path.join('./results', f'{i:03d}.png'), save)
        #--------------------------------------------------------------------#
                    
    # Normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # for i in range(len(render_imgs)):
    #     render_imgs_[i,:,:,:] = Normalize(render_imgs_[i,:,:,:])

    return render_imgs_


def GMSampler(dist_pool_mu, dist_pool_sigma, batch_size, clean_imgs=None, split='train', flag='AT'):
    args = get_opts()
    """
    从对应类别的混合高斯分布池中采样视角参数，并生成对应的渲染对抗样本
    Args:
        label: a batch of true label
        dist_pool: The uploaded GM distribution pool
        batch_size: num of adversarial viewpoint renderings

    Returns: a batch of adversarial viewpoint rendering images

    """
    """
    dist_pool以四维数组形式存储 m为标签（数字表示） 后三维为对应的 n*k*6矩阵(n为每个物体设置采集的分布数,k为一个分布的分量数)
    
    dist_pool_mu: array(m,n,k,6)
    dist_pool_sigma: array(m,n,k,6)
    dist_pool_omiga: array(m,n,k,6) // omiga = 1.0/K
    """
    M = dist_pool_mu.shape[0]
    # N = dist_pool_mu.shape[1]
    k = dist_pool_mu.shape[2]

    if split == 'train':
        ckpt_path = './ckpts/nerf/train/'
    else:
        ckpt_path = './ckpts/nerf/test/'

    label_list = os.listdir(ckpt_path)
    label_list.sort()

    label_idx = np.random.randint(low=0, high=M, size=batch_size)  # 生成随机的label
    # n_idx = np.random.randint(low=0, high=N, size=batch_size)

    sample_all = np.zeros([batch_size, 6])

    a = [30, 180, 70, 1.0, 0.5, 0.5]
    b = [0, 0, 0, 4.0, 0, 0]
    n_idxs = []
    for i in range(batch_size):
        
        N = len(os.listdir(ckpt_path+label_list[label_idx[i]]+'/'))
        n_idx = np.random.randint(low=0, high=N, size=1)
        n_idxs.append(n_idx)
        # print(n_idx)
        if args.share_dist:
            mu = dist_pool_mu[label_idx[i], 0, :, :].squeeze()
            sigma = dist_pool_sigma[label_idx[i], 0, :, :].squeeze()
        else:
            mu = dist_pool_mu[label_idx[i], n_idx, :, :].squeeze()
            sigma = dist_pool_sigma[label_idx[i], n_idx, :, :].squeeze()

        F = np.random.choice(a=np.arange(k), size=6, replace=True, p=np.ones(k)/k)
        sample = np.zeros(6)
        # print(mu)
        # print(sigma)
        for j in range(6):
            L = int(F[j])
            sample[j] = a[j]*np.tanh(np.random.normal(loc=mu[L, j], scale=sigma[L, j], size=1))+b[j]   # 得到一个viewpoint的一组参数

        sample_all[i, :] = sample # 加进总的viewpoints中


    labels = np.zeros(batch_size)
    for i in range(batch_size):
        labels[i] = int(label_list[label_idx[i]])
    labels = labels.astype(int)
    labels = torch.from_numpy(labels)
    render_imgs = render_image(sample_all, labels, n_idxs, split=split) # 遍历渲染第M个类的第N个物体
    render_imgs = AddBackground(render_imgs, clean_imgs, batch_size, split=split, flag=flag)

    return  render_imgs, labels



def ViewFoolSampler(dist_pool_mu, dist_pool_sigma, batch_size, clean_imgs=None, split='train', flag='AT'):
    """
    执行ViewFool攻击后，从单高斯分布中采样，并生成对应的渲染对抗样本
    Args:
        label: a batch of true label
        dist_pool: The uploaded GM distribution pool
        batch_size: num of adversarial viewpoint renderings

    Returns: a batch of adversarial viewpoint rendering images

    """
    """
    viewfool 的dist_pool以三维数组形式存储 m为标签（数字表示） 后二维为对应的 n*6矩阵(n为每个物体设置采集的分布数,k为一个分布的分量数)
    
    dist_pool_mu: array(m,n,6)
    dist_pool_sigma: array(m,n,6)
    """
    M = dist_pool_mu.shape[0]
    # N = dist_pool_mu.shape[1]

    if split == 'train':
        ckpt_path = './ckpts/nerf/train/'
    else:
        ckpt_path = './ckpts/nerf/test/'

    label_list = os.listdir(ckpt_path)
    label_list.sort()

    label_idx = np.random.randint(low=0, high=M, size=batch_size)  # 生成随机的label
    # n_idx = np.random.randint(low=0, high=N, size=batch_size)

    sample_all = np.zeros([batch_size, 6])

    a = [30, 180, 70, 1.0, 0.5, 0.5]
    b = [0, 0, 0, 4.0, 0, 0]
    n_idxs = []
    for i in range(batch_size):
        
        N = len(os.listdir(ckpt_path+label_list[label_idx[i]]+'/'))
        n_idx = np.random.randint(low=0, high=N, size=1)
        n_idxs.append(n_idx)
        # print(n_idx)
        mu = dist_pool_mu[label_idx[i], n_idx, :].squeeze()
        sigma = dist_pool_sigma[label_idx[i], n_idx, :].squeeze()
        sample = np.zeros(6)
        # print(mu)
        # print(sigma)
        for j in range(6):
            sample[j] = a[j]*np.tanh(np.random.normal(loc=mu[j], scale=sigma[j], size=1))+b[j]   # 得到一个viewpoint的一组参数

        sample_all[i, :] = sample # 加进总的viewpoints中


    labels = np.zeros(batch_size)
    for i in range(batch_size):
        labels[i] = int(label_list[label_idx[i]])
    labels = labels.astype(int)
    labels = torch.from_numpy(labels)
    render_imgs = render_image(sample_all, labels, n_idxs, split=split) # 遍历渲染第M个类的第N个物体
    render_imgs = AddBackground(render_imgs, clean_imgs, batch_size, split=split, flag=flag)

    return  render_imgs, labels


def RandomSampler(batch_size, clean_imgs=None, split='test'):

    ckpt_path = './ckpts/nerf/test/'
    label_list = os.listdir(ckpt_path)
    label_list.sort()
    M = len(label_list)

    label_idx = np.random.randint(low=0, high=M, size=batch_size)  # 生成随机的label
    # n_idx = np.random.randint(low=0, high=N, size=batch_size)

    sample_all = np.zeros([batch_size, 6])
    random_type = 'side'

    if random_type=='side':
        a = [0, 360, -10, 2.0, 1.0, 1.0]
        b = [0, -180, 70, 3.0, -0.5, -0.5]
    elif random_type=='up':
        a = [0, 360, -30, 2.0, 1.0, 1.0]
        b = [0, -180, 15, 3.0, -0.5, -0.5]
    elif random_type=='full':
        a = [60, 360, 140, 2.0, 1.0, 1.0]
        b = [-30, -180, -70, 3.0, -0.5, -0.5]

    n_idxs = []
    for i in range(batch_size):
        
        N = len(os.listdir(ckpt_path+label_list[label_idx[i]]+'/'))
        n_idx = np.random.randint(low=0, high=N, size=1)
        n_idxs.append(n_idx)
        # print(n_idx)

        sample = np.zeros(6)
        # print(mu)
        # print(sigma)
        for j in range(6):
            sample[j] = a[j]*np.random.random(1)+b[j]   # 得到一个viewpoint的一组参数

        sample_all[i, :] = sample # 加进总的viewpoints中


    labels = np.zeros(batch_size)
    for i in range(batch_size):
        labels[i] = int(label_list[label_idx[i]])
    labels = labels.astype(int)
    labels = torch.from_numpy(labels)
    render_imgs = render_image(sample_all, labels, n_idxs) # 遍历渲染第M个类的第N个物体
    render_imgs = AddBackground(render_imgs, clean_imgs, batch_size, split=split)

    return  render_imgs, labels



# ------------------------------------------------------------------------------------------------------------- #


# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
# gpus = [0, 1, 2, 3]

kwargs = {'num_workers': 10*len(gpus), 'pin_memory': True} if use_cuda else {}

# setup data loader

traindir = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/imagenet/train'
valdir = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/imagenet/val'

train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                                      std=[0.229, 0.224, 0.225])
        ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                                      std=[0.229, 0.224, 0.225])
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)



def train(args, model, device, train_loader, optimizer, epoch, dist_pool_mu, dist_pool_sigma):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        if args.train_mood == 'AT':
            render_imgs, render_label = RandomSampler(batch_size=16)
            # print('img label:' , target)
            # print('render label:' , render_label)
            # print('img data:' , data.shape)
            # print('render data:' , render_imgs.shape)

            data = torch.cat([data, render_imgs], dim=0)
            target = torch.cat([target, render_label], dim=0)
            # 打乱数据
            data = data.numpy()
            target = target.numpy()
            index = np.arange(len(target))
            np.random.shuffle(index)
            data = data[index]
            target = target[index]

            data = torch.from_numpy(data)
            target = torch.from_numpy(target)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        # calculate robust loss
        logits = model(data)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Clean sample in Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Clean sample in Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_robust_last(model, device):
    model.eval()
    train_loss = 0
    correct = 0
    data_num = 0

    NES_GMM_search(model, mood='init')
    # load dist_pool
    dist_pool_mu = np.load(f'./dist_pool/dist_pool_mu_{args.AT_exp_name}.npy')
    dist_pool_sigma = np.load(f'./dist_pool/dist_pool_sigma_{args.AT_exp_name}.npy')
        
    for i in range(10):
        data, target = GMSampler(dist_pool_mu, dist_pool_sigma, args.batch_size, split='test')
        data, target = data.to(device), target.to(device)
        data_num += len(data)

        output = model(data)
        train_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= data_num

    print('render train set robust_accuracy: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, data_num,
        100. * correct / data_num))
    training_accuracy = correct / data_num

    return train_loss, training_accuracy


def eval_robust_train(model, device, dist_pool_mu, dist_pool_sigma, flag='AT'):
    model.eval()
    train_loss = 0
    correct = 0
    data_num = 0
    with torch.no_grad():
        for i in range(10):
            data, target = GMSampler(dist_pool_mu, dist_pool_sigma, args.batch_size, split='train', flag=flag)
            data, target = data.to(device), target.to(device)
            data_num += len(data)

            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= data_num

    print('GMFool trainset accuracy: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, data_num,
        100. * correct / data_num))
    training_accuracy = correct / data_num

    return train_loss, training_accuracy


def eval_robust_test(model, device):
    model.eval()
    train_loss = 0
    correct = 0
    data_num = 0

    NES_GMM_search(model, mood='eval_test')
    # load dist_pool
    dist_pool_mu = np.load(f'./dist_pool/dist_pool_mu_{args.AT_exp_name}_eval_test.npy')
    dist_pool_sigma = np.load(f'./dist_pool/dist_pool_sigma_{args.AT_exp_name}_eval_test.npy')

    with torch.no_grad():
        for i in range(5):
            data, target = GMSampler(dist_pool_mu, dist_pool_sigma, args.batch_size, split='test')
            data, target = data.to(device), target.to(device)
            data_num += len(data)

            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= data_num

    print('GMFool testset accuracy: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, data_num,
        100. * correct / data_num))
    training_accuracy = correct / data_num

    return train_loss, training_accuracy


def eval_viewfool_test(model, device):
    model.eval()
    train_loss = 0
    correct = 0
    data_num = 0

    NES_viewfool_search(model, mood='test')
    # load dist_pool
    dist_pool_mu = np.load(f'./dist_pool/dist_pool_mu_{args.AT_exp_name}_viewfool_test.npy')
    dist_pool_sigma = np.load(f'./dist_pool/dist_pool_sigma_{args.AT_exp_name}_viewfool_test.npy')

    with torch.no_grad():
        for i in range(5):
            data, target = ViewFoolSampler(dist_pool_mu, dist_pool_sigma, args.batch_size, split='test')
            data, target = data.to(device), target.to(device)
            data_num += len(data)

            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= data_num

    print('ViewFool testset accuracy: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, data_num,
        100. * correct / data_num))
    training_accuracy = correct / data_num

    return train_loss, training_accuracy


def eval_random(model, device):
    model.eval()
    train_loss = 0
    correct = 0
    data_num = 0
    with torch.no_grad():
        for i in range(10):
            data, target = RandomSampler(args.batch_size)
            data, target = data.to(device), target.to(device)
            data_num += len(data)

            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= data_num

    print('Random render accuracy: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, data_num,
        100. * correct / data_num))
    training_accuracy = correct / data_num

    return train_loss, training_accuracy



def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    # if epoch <= 9:
    #     lr = args.lr + 0.001
    if epoch >= 35:
        lr = args.lr * 0.1
    if epoch >= 65:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#----------------------------------------------------------------------------------------------------------#

def main():
    # init model, ResNet18() can be also used here for training
    model = torchvision.models.resnet50(pretrained=False)
    # checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/NeRF/ckpts/resnet50-0676ba61.pth'
    checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/model-imagenet-100-ckpts/model-res50AT-test5_natural-epoch40.pt'
    num_ftr = model.fc.in_features
    model.fc = nn.Linear(num_ftr,100)

    model.load_state_dict(torch.load(checkpoint))
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(41, args.epochs + 41):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        
        if args.train_mood == 'AT':
            # random attack
            
             
            # warm_start if epoch>1
            if epoch == 0:
                # attack to generate dist_pool first
                print('================================================================')
                eval_random(model, device)
                print('================================================================')
                
                print('================================================================')
                eval_viewfool_test(model, device)
                print('================================================================')

            else:
                # adversarial training
                train(args, model, device, train_loader, optimizer, epoch, dist_pool_mu=None, dist_pool_sigma=None)

                # evaluation on natural examples
                print('================================================================')
                eval_train(model, device, train_loader)
                eval_test(model, device, test_loader)
                if args.train_mood == 'AT':
                # eval_robust_train(model, device, dist_pool_mu, dist_pool_sigma, flag='before_AT')
                    if epoch % 5 == 0:
                        eval_viewfool_test(model, device)
                print('================================================================')
                
        else:
            dist_pool_mu = None
            dist_pool_sigma = None

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(), os.path.join(model_dir, 'model-res50AT-{}-epoch{}.pt'.format(args.AT_exp_name, epoch)))
            # torch.save(optimizer.state_dict(), os.path.join(model_dir, 'opt-res50-checkpoint_epoch{}.tar'.format(epoch)))
    
    # GMFool ATTACK
    # eval_robust_last(model, device)

if __name__ == '__main__':
    main()
