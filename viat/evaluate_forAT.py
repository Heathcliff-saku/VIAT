from rendering_image_forAT import render_image
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

def metric(prediction, label, target_label=0, target_flag=False):
    loss_func = nn.CrossEntropyLoss()
    if target_flag == False:
        # 对于无目标攻击，loss值越大，代表攻击越成功
        loss = loss_func(prediction, label)
    else:
        # 对于无目标攻击，loss的负值越大，说明loss越小，越接近，代表攻击越成功
        loss = - loss_func(prediction, target_label)
    return loss


@torch.no_grad()
def comput_fitness(model, label, ckpt_path, solution, is_viewfool):
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
        x = render_image(solution, ckpt_path, is_viewfool=is_viewfool) # ndarray [N,W,H,C]

    x = np.array(x)

    tensors = torch.zeros(x.shape[0], x.shape[3], 224, 224)
    for i in range(len(x)):
        img = x[i, :]
        img = Image.fromarray(img)
        tensor = transform(img)
        tensors[i,:,:,:] = tensor

    tensors = tensors.cuda()


    model.eval()
    with torch.no_grad():
        # 得到预测的softmax向量
        prediction = model(tensors)

    # true_label = np.zeros((1, 1000))
    # true_label[:, label] = 1.0
    # true_label = torch.from_numpy(true_label)

    label = torch.LongTensor([label])
    # print('current class:', label)

    rewards = []
    for i in range(x.shape[0]):
        reward = metric(prediction[i].unsqueeze(0), label=label.cuda())
        reward = reward.cpu().detach().numpy()
        rewards += [reward]
    # loss_var = compute_ver(sigma, solution)
    # print('分类损失：', reward)
    # print('高斯熵损失：', 0.5*loss_var)

    return rewards





