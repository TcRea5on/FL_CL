#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid_2
from utils.options import args_parser
from models.Update import LocalUpdate, globalupdate
from models.Nets import MLP, CNNMnist, CNNCifar, fed_contrastive,fed_contrastive_2
from models.Fed import FedAvg
from models.test import test_img
from torch.utils.data import DataLoader, Dataset
from models.test import test_user
import InfoNCE2


def cal_feature(image, net_list_fed):
    features = {}
    for i in range(len(net_list_fed)):
        # net_list_fed[i].load_state_dict(net_list_local[i].state_dict(),strict=False)
        features[i] = net_list_fed[i].forward(image.to(args.device))
    return features

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

if __name__ == '__main__':
    # parse args
    args = args_parser()
    #print(args.gpu)
    args.iid = False
    #print(torch.cuda.is_available())
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #print(args.device)
    args.dataset = 'cifar'
    print(args.num_users)

    # load dataset and split users

    print(time.asctime(time.localtime(time.time())))

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(1, 1, 1))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        # 35行到39行是按照标签收集一些收据，之后构建一组对比数据
        number_list = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        for i in range(60000):
            # print(dataset_train[i])
            number = dataset_train[i][1]
            # print(number)
            # print(number_list[number])
            number_list[number].append(i)
        #print(number_list)


        print('Non-iid')
        dict_users_train,dict_users_test = mnist_noniid(dataset_train,dataset_test, args.num_users)
        #for i in range(len(dict_users)):
            #print(len(dict_users[i]))


        #for i in range(len(dict_users[16])):
            #print(dataset_train[dict_users[16][i]])
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        #print(len(dataset_test))

        class_list = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        for i in range(50000):
            # print(dataset_train[i])
            number = dataset_train[i][1]
            # print(number)
            # print(number_list[number])
            class_list[number].append(i)


        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            print('non-iid')
            dict_users_train,dict_users_test = cifar_noniid_2(dataset_train,dataset_test, args.num_users)
            print(len(dict_users_test[1]))
            #dict_users_test = cifar_noniid(dataset_test,args.num_users,200,50)
        for i in range(len(dict_users_train[6])):
            print(dataset_train[dict_users_train[6][i]])
        print('------------------------------------------------------')
        for i in range(len(dict_users_test[6])):
            print(dataset_test[dict_users_test[6][i]])
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        net_fed = fed_contrastive_2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        print('cnn')
        # 初始化global网络，fed_contrastive类在Nets里，是global环节利用SimCLR更新CNN参数的网络
        net_fed = fed_contrastive(args=args).to(args.device)
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        print('mlp')
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()
    net_fed.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        print(len(w_locals))
    else:
        print('not all users')

    global_data_1 = []
    global_data_2 = []

    if (args.dataset == 'mnist'):
        for i in range(10):
            for j in range(10):
                global_data_1.extend([number_list[j][i]])
        for i in range(125, 250):
            for j in range(10):
                global_data_2.extend([number_list[j][i]])

    if (args.dataset == 'cifar'):
        for i in range(10):
            for j in range(10):
                global_data_1.extend([class_list[j][i]])
        for i in range(125, 250):
            for j in range(10):
                global_data_2.extend([class_list[j][i]])

    #global_images_1 = DataLoader(DatasetSplit(dataset_train, global_data_1), batch_size=50, shuffle=False)

    # another data collection
    '''
    for i in range(100):
        for j in range(10):
            global_data_1.extend(number_list[j][i])
    for i in range(100,200):
        for j in range(10):
            global_data_2.extend(number_list[j][i])
    '''

    local_class = {}
    local_net = {}
    net_para = {}
    #local_fed_class = {}
    local_fed_net = {}
    local_fed_para = {}

    if(args.dataset=='mnist'):
        for i in range(args.num_users):
            local_class[i] = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[i])
            local_net[i] = CNNMnist(args=args).to(args.device)
            local_net[i].train()
            net_para[i] = local_net[i].state_dict()
            local_fed_net[i] = fed_contrastive(args=args).to(args.device)
            local_fed_net[i].train()
            local_fed_para[i] = local_fed_net[i].state_dict()

    if (args.dataset == 'cifar'):
        for i in range(args.num_users):
            local_class[i] = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[i])
            local_net[i] = CNNCifar(args=args).to(args.device)
            local_net[i].train()
            net_para[i] = local_net[i].state_dict()
            local_fed_net[i] = fed_contrastive_2(args=args).to(args.device)
            local_fed_net[i].train()
            local_fed_para[i] = local_fed_net[i].state_dict()


    for iter in range(args.epochs):
        loss_locals = []
        acc_locals = []
        #if not args.all_clients:
            #w_locals = []
        m = args.num_users#max(int(args.frac * args.num_users), 1)
        idxs_users = np.arange(0,args.num_users)
        #print(idxs_users)


        for idx in idxs_users:
            print(idx)
            local = local_class[idx]#LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            net_para[idx], loss,acc = local.train(net=local_net[idx].to(args.device))
            local_net[idx].load_state_dict(net_para[idx])
            local_fed_net[idx].load_state_dict(net_para[idx],strict=False)
            #if args.all_clients:
                #w_locals[idx] = copy.deepcopy(w)
            #else:
                #w_locals.append(copy.deepcopy(w))

            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))



        #global part
        #complexity calculate

        if(iter<args.epochs-1):
            for i in range(args.num_users):
                #print(i)
                global_update = globalupdate(args=args, dataset=dataset_train, idxs_1=global_data_1, idxs_2=global_data_2)
                local_fed_para[i] = global_update.train_2(local_fed_net,i,args.num_users)
            for i in range(args.num_users):
                local_fed_net[i].load_state_dict(local_fed_para[i])
                local_net[i].load_state_dict(local_fed_para[i],strict=False)



            buffer_1 = global_data_1[0:10]
            for i in range(9):
                global_data_1[i * 10:i * 10 + 10] = global_data_1[i * 10 + 10:i * 10 + 20]
            global_data_1[-10:] = buffer_1

        users_acc = []
        for i in range(args.num_users):
            acc = test_user(local_net[i], dataset_test, dict_users_test[i], args)
            users_acc.append(acc)
        average_acc = sum(users_acc) / args.num_users
        print(average_acc)



        # If_CimCLR = 0

        # 将经过fedavg和SimCLR更新之后的net_glob记录，分发给所有Local
        # net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        acc_avg = sum(acc_locals) / len(acc_locals)
        print('Round {:3d}, Average acc {:.3f}'.format(iter, acc_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    users_acc = []
    for i in range(args.num_users):
        acc = test_user(local_net[i],dataset_test,dict_users_test[i],args)
        users_acc.append(acc)
    average_acc = sum(users_acc)/args.num_users
    print(average_acc)



    '''
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print(time.asctime(time.localtime(time.time())))
    '''