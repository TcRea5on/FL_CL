#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid_2
from utils.options import args_parser
from models.Update import LocalUpdate,globalupdate
from models.Nets import MLP, CNNMnist, CNNCifar,fed_contrastive,fed_contrastive_2,CNNCifar_2,fed_contrastive_3
from models.Fed import FedAvg
from models.test import test_img
from torch.utils.data import DataLoader, Dataset
import InfoNCE2


if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args.all_clients)
    print(args.gpu)
    print(torch.cuda.is_available())
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    args.dataset = 'mnist'
    print(args.model)
    args.iid = False

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(1,1,1))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        # 35行到39行是按照标签收集一些收据，之后构建一组对比数据
        number_list = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        for i in range(8000):
            #print(dataset_train[i])
            number = dataset_train[i][1]
            #print(number)
            #print(number_list[number])
            if (len(number_list[number]) < 500):
                number_list[number].append(i)
        print(number_list)


        #else:
        print('Non-iid')
        dict_users_train,dict_users_test = mnist_noniid(dataset_train,dataset_test, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)


        class_list = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        for i in range(50000):
            number = dataset_train[i][1]
            class_list[number].append(i)


        #print(dataset_train[0][1])
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            print('non-iid')
            dict_users_train,dict_users_test = cifar_noniid_2(dataset_train,dataset_test,args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        net_fed = fed_contrastive_2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        #初始化global网络，fed_contrastive类在Nets里，是global环节利用SimCLR更新CNN参数的网络
        net_fed = fed_contrastive(args=args).to(args.device)
        net_glob_pre = CNNMnist(args=args).to(args.device)
        net_glob_new = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
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

    #global端两组用于对比的数据
    if(args.dataset == 'mnist'):
        for i in range(25):
            for j in range(10):
                global_data_1.extend([number_list[j][i]])
        for i in range(25,50):
            for j in range(10):
                global_data_2.extend([number_list[j][i]])

    if(args.dataset == 'cifar'):
        for i in range(100):
            for j in range(10):
                global_data_1.extend([class_list[j][i]])
        for i in range(100,200):
            for j in range(10):
                global_data_2.extend([class_list[j][i]])

    #another data collection
    '''
    for i in range(100):
        for j in range(10):
            global_data_1.extend(number_list[j][i])
    for i in range(100,200):
        for j in range(10):
            global_data_2.extend(number_list[j][i])
    '''


    #开始训练
    for iter in range(args.epochs):
        loss_locals = []
        acc_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        '''
        #net_fed记录global参数，对参数利用infoNCE更新
        net_fed.load_state_dict(w_glob, strict=False)
        global_test = globalupdate(args=args, dataset=dataset_train, idxs_1=global_data_1, idxs_2=global_data_2,iter=iter)
        global_test.train(net_fed)
        net_glob.load_state_dict(net_fed.state_dict(), strict=False)
        '''

        for idx in idxs_users:
            #本地更新
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            w, loss,acc= local.train(iter,net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))

        #得到fedavg
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))


        net_fed.load_state_dict(w_glob,strict=False)
        #在fedavg的基础上优化CNN层
        if(iter<args.epochs-1):
            global_test = globalupdate(args = args,dataset=dataset_train,idxs_1=global_data_1,idxs_2=global_data_2)
            global_test.train(net_fed,iter)
            net_glob.load_state_dict(net_fed.state_dict(), strict=False)

            #改变一下用于对比数据的顺序，让没有做过正例的数据移动到矩阵中正例的位置
            buffer_1 = global_data_1[0:10]
            for i in range(24):
                global_data_1[i * 10:i * 10 + 10] = global_data_1[i * 10 + 10:i * 10 + 20]
            global_data_1[-10:] = buffer_1

            buffer_2 = global_data_2[0:10]
            for i in range(24):
                global_data_2[i * 10:i * 10 + 10] = global_data_2[i * 10 + 10:i * 10 + 20]
            global_data_2[-10:] = buffer_2




        # 将经过fedavg和SimCLR更新之后的net_glob记录，分发给所有Local
        #net_glob.load_state_dict(w_glob)
        '''
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        '''

        # print loss

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        acc_avg = sum(acc_locals) / len(acc_locals)
        print('Round {:3d}, Average acc {:.3f}'.format(iter, acc_avg))
        #loss_train.append(loss_avg)

    # plot loss curve
    '''
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    '''

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))