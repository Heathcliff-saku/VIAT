from rendering_image import render_image
import numpy as np
import cv2 as cv
from PIL import Image
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from datasets.opts import get_opts
import time
'''
    定义NES每个方案的适应度
'''

def metric(prediction, label, target_label, target_flag):
    loss_func = nn.CrossEntropyLoss()
    if target_flag == False:
        # 对于无目标攻击，loss值越大，代表攻击越成功
        loss = loss_func(prediction, label)
    else:
        # 对于无目标攻击，loss的负值越大，说明loss越小，越接近，代表攻击越成功
        loss = - loss_func(prediction, target_label)
    return loss


def compute_ver(sigma, mu, num_sample=1000):
  # 计算多元高斯分布熵

  random = np.zeros([num_sample, 6])
  gamma = np.random.normal(loc=mu[0], scale=sigma[0], size=num_sample)
  th = np.random.normal(loc=mu[1], scale=sigma[1], size=num_sample)
  phi = np.random.normal(loc=mu[2], scale=sigma[2], size=num_sample)
  r = np.random.normal(loc=mu[3], scale=sigma[3], size=num_sample)
  a = np.random.normal(loc=mu[4], scale=sigma[4], size=num_sample)
  b = np.random.normal(loc=mu[5], scale=sigma[5], size=num_sample)
  random[:, 0] = gamma
  random[:, 1] = th
  random[:, 2] = phi
  random[:, 3] = r
  random[:, 4] = a
  random[:, 5] = b
  mu = random.mean(axis=0)
  var = (random - mu).T @ (random - mu) / random.shape[0]

  loss_var = - np.log(np.linalg.det(var))
  loss_var = 0.03 * loss_var
  return loss_var

@torch.no_grad()
def comput_fitness(solution):
    '''

    Args:
        solution: 当前采样到的参数值 (th, phi, r)
    Returns:
        reward: 适应度值
    '''
    args = get_opts()

    # 利用求得的参数渲染一幅图像
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        x = render_image(solution) # ndarray [N,W,H,C]

    x = np.array(x)

    tensors = torch.zeros(x.shape[0], x.shape[3], 224, 224)
    for i in range(len(x)):
        img = x[i, :]
        img = Image.fromarray(img)
        tensor = transform(img)
        tensors[i,:,:,:] = tensor

    # print(tensors.shape)
    # print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (c,h,w)
    # tensor = torch.unsqueeze(tensor, 0)  # 返回一个新的tensor,对输入的既定位置插入维度1
    # print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (1,c,h,w)
    tensors = tensors.cuda()

    model = models.resnet50(pretrained=False)
    checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/NeRF/ckpts/resnet50-0676ba61.pth'

    # model = models.inception_v3(pretrained=False)
    # checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/NeRF/ckpts/inception_v3_google-0cc3c7bd.pth'

    #model = models.vit_b_16(pretrained=False)
    #checkpoint = '/data/home/scv7303/run/rsw_/NeRFAttack/NeRF/ckpts/vit_b_16-c867db91.pth'

    model.load_state_dict(torch.load(checkpoint))
    model.cuda()

    model.eval()
    with torch.no_grad():
        # 得到预测的softmax向量
        prediction = model(tensors)

    true_label = np.zeros((1, 1000))
    true_label[:, args.label] = 1.0
    true_label = torch.from_numpy(true_label)

    target_label = np.zeros((1, 1000))
    """
    584: hair slide
    650: microphone, mike
    """

    target_label[:, args.target_label] = 1.0
    target_label = torch.from_numpy(target_label)

    rewards = []
    for i in range(x.shape[0]):
        reward = metric(prediction[i].unsqueeze(0), label=true_label.cuda(), target_label=target_label, target_flag=args.target_flag)
        reward = reward.cpu().detach().numpy()
        rewards += [reward]
    # loss_var = compute_ver(sigma, solution)
    # print('分类损失：', reward)
    # print('高斯熵损失：', 0.5*loss_var)

    return rewards





