#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from models.Nets import fed_contrastive
import matplotlib
import matplotlib.pyplot as plt
from InfoNCE2 import ContrastiveLoss




class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.idxs = idxs
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self,epoch,net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr*(0.5 ** int(epoch / 10)), momentum=self.args.momentum)
        #lr:*(0.5 ** int(epoch / 10))

        #print(len(self.idxs))
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_corrects = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)


                batch_correct = (torch.sum(torch.argmax(log_probs, dim=1) == labels)).item()
                batch_corrects.append(batch_correct)
                #print(batch_correct)
                #print(torch.argmax(log_probs,dim=1))
                #print(labels)
                #print(torch.sum(torch.argmax(log_probs,dim=1)==labels))

                #print(log_probs.size())
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            #print(batch_corrects)
            #print(sum(batch_corrects))
            train_acc = sum(batch_corrects)/len(self.idxs)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),train_acc


class globalupdate(object):
    '''
    现在这个代码只是一个尝试，并没有完全实现我们之前的构想
    大概思路写在这里
    首先构造一对数据集，这两个数据集之间没有重复数据，但是标签相互对应
    然后还是fedavg，会初步更新出一个net_glob的参数
    然后我们再利用对比学习和刚才的两个数据集更新这个net_glob的CNN参数
    更新完以后再分发个各个local，进入下一轮迭代
    '''


    def __init__(self, args, dataset = None, idxs_1 = None,idxs_2 = None,iter= None):
        self.args = args
        self.iter = iter
        #构造成对数据集
        self.data_test_1 = DataLoader(DatasetSplit(dataset, idxs_1), batch_size=50, shuffle=False)
        self.data_test_2 = DataLoader(DatasetSplit(dataset, idxs_2), batch_size=50, shuffle=False)


    def train(self,net,epoch):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.g_lr*(0.5**int(epoch/5)), momentum=self.args.momentum)
        #
        print(self.args.g_lr*(0.5**int(epoch/5)))
        epoch_loss = []
        imagess_1 = []
        imagess_2 = []
        SimCLR_loss = ContrastiveLoss(50,0.5)
        for iter in range(self.args.global_ep):
            for batch_idx_1,(images_1, labels_1) in enumerate(self.data_test_1):
                images_1 = images_1.to(self.args.device)
                #print(images_1.size())
                imagess_1.append(images_1)
            for batch_idx_2, (images_2, labels_2) in enumerate(self.data_test_2):
                images_2 = images_2.to(self.args.device)
                # print(images_1.size())
                imagess_2.append(images_2)
            net.zero_grad()
            for k in range(len(imagess_1)):
                emb_i = net.forward(imagess_1[k].to(self.args.device))
                emb_j = net.forward(imagess_2[k].to(self.args.device))
                loss = SimCLR_loss.forward(emb_i, emb_j)
                loss.backward()
                optimizer.step()
            #print('-------------------------------------------------------------------------------------------------')
            '''
            for batch_idx_2,(images_2, labels_2) in enumerate(self.data_test_2):
                images_2 = images_2.to(self.args.device)
                emb_j.extend(images_2)
                net.zero_grad()
                emb_i = net.forward(images_1)
                emb_j = net.forward(images_2)
                loss = SimCLR_loss.forward(emb_i,emb_j)
                print(loss)
                loss.backward()
                optimizer.step()
                '''
    def cal_feature(self,image,net_list_fed):
        features = {}
        for i in range(len(net_list_fed)):
            #net_list_fed[i].load_state_dict(net_list_local[i].state_dict(),strict=False)
            features[i] = net_list_fed[i].forward(image.to(self.args.device))
        return  features

    #def cal_sum_loss(self,key,):




    def train_2(self,fed_net_list,key,num_users):
        #net_list[key].train()
        optimizer = torch.optim.SGD(fed_net_list[key].parameters(), lr=0.001, momentum=self.args.momentum)
        SimCLR_loss = ContrastiveLoss(50, 0.5)

        for batch_idx_1,(images_1, labels_1) in enumerate(self.data_test_1):
            images_1 = images_1.to(self.args.device)
            features = self.cal_feature(images_1,fed_net_list)
            #print(features)
            fed_net_list[key].zero_grad()
            loss = 0
            contrastive_user = np.random.choice(range(num_users),20,replace=False)
            for j in contrastive_user:
                loss = loss+SimCLR_loss.forward(features[key],features[j])
            if(key in contrastive_user):
                loss = loss - SimCLR_loss(features[key],features[key])
            loss.backward()
            optimizer.step()
        return fed_net_list[key].state_dict()



    def train_2_plus(self, fed_net_list, key,features):
        # net_list[key].train()
        optimizer = torch.optim.SGD(fed_net_list[key].parameters(), lr=0.001, momentum=self.args.momentum)
        SimCLR_loss = ContrastiveLoss(50, 0.5)
        # print(features)
        fed_net_list[key].zero_grad()
        loss = 0
        contrastive_user = np.random.choice(range(100), 10,replace=False)
        for j in contrastive_user:
            loss = loss + SimCLR_loss.forward(features[key], features[j])
        #loss = loss - SimCLR_loss(features[key], features[key])
        loss.backward()
        optimizer.step()
        return fed_net_list[key].state_dict()


    def cal_loss(self,fed_net_list,images_1):
        SimCLR_loss = ContrastiveLoss(50, 0.5)
        loss = np.zeros((100,100))
        #for batch_idx_1, (images_1, labels_1) in enumerate(self.data_test_1):
        features = self.cal_feature(images_1, fed_net_list)
        for i in range(100):
            for j in range(i+1,100):
                loss[i][j] = SimCLR_loss(features[i],features[j])
                loss[j][i] = loss[i][j]
        return loss











    '''
    def cal_feature(self,net):
        i = 0
        for batch_idx_1,(images_1, labels_1) in enumerate(self.data_test_1):
            #print(images.size())
            images, labels = images_1.to(self.args.device), labels_1.to(self.args.device)
            #print('-------------------------------------------------------------------------------')
            return net.forward(images)
    '''





    #def SimCLR_loss(self):

