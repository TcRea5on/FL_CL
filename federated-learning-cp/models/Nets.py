#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #print(x.shape)
        return x

class fed_contrastive(nn.Module):
    #输入数据集会输出数据集里的每一个图片的40维的表征
    def __init__(self, args):
        super(fed_contrastive, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1_g = nn.Linear(320, 100)
        self.fc2_g = nn.Linear(100, 30)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        #print(x.size())
        x = F.relu(self.fc1_g(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2_g(x)
        #print(x)
        #print(x.size())
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class fed_contrastive_2(nn.Module):
    def __init__(self, args):
        super(fed_contrastive_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1_g = nn.Linear(16 * 5 * 5, 120)
        self.fc2_g = nn.Linear(120, 100)
        self.fc3_g = nn.Linear(100, 30)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1_g(x))
        x = F.relu(self.fc2_g(x))
        x = self.fc3_g(x)
        return x



class CNNCifar_2(nn.Module):
    def __init__(self, args):
        super(CNNCifar_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

    def forward(self, x):
        # 输入x -> conv1 -> relu -> 2x2窗口的最大池化
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # 输入x -> conv2 -> relu -> 2x2窗口的最大池化
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class fed_contrastive_3(nn.Module):
    def __init__(self, args):
        super(fed_contrastive_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1_g = nn.Linear(16 * 5 * 5, 120)
        self.fc2_g = nn.Linear(120, 100)
        self.fc3_g = nn.Linear(100, 30)

    def forward(self, x):
        # 输入x -> conv1 -> relu -> 2x2窗口的最大池化
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # 输入x -> conv2 -> relu -> 2x2窗口的最大池化
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1_g(x))
        x = F.relu(self.fc2_g(x))
        x = self.fc3_g(x)
        return x





