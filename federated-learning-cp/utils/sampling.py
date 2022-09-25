#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import random
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset_train,dataset_test, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs_train,num_imgs_test = 100,600,100
    idx_shard = [i for i in range(num_shards)]#train,test dataset share one idx_shard
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs_train = np.arange(num_shards*num_imgs_train)#dataset_train index
    idxs_test = np.arange(num_shards*num_imgs_test)#dataset_test index
    labels_train = dataset_train.train_labels.numpy()
    labels_test = dataset_test.train_labels.numpy()
    #print(labels)

    # sort labels
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_train = idxs_labels_train[:,idxs_labels_train[1,:].argsort()]
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_train = idxs_labels_train[0,:]
    idxs_test = idxs_labels_test[0,:]
    #print(idxs)
    #print(dataset[11924])

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate((dict_users_train[i], idxs_train[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[rand*num_imgs_test:(rand + 1)*num_imgs_test]), axis=0)
    return dict_users_train,dict_users_test


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid_2(dataset, num_users,number_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    #int(2000*((1-0.2)*np.random.random()+0.2))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # divide and assign
    for i in range(num_users):
        j = int(i/10)
        i_num =int(2000*((1-0.2)*np.random.random()+0.2))
        domain_class_1 = set(np.random.choice(number_list[j*2],int(i_num*0.4),replace=False))
        domain_class_2 = set(np.random.choice(number_list[j*2+1],int(i_num*0.4),replace=False))
        for rand in domain_class_1:
            dict_users[i] = np.concatenate((dict_users[i], [rand]), axis=0)
        for rand in domain_class_2:
            dict_users[i] = np.concatenate((dict_users[i], [rand]), axis=0)
        for k in range(10):
            if(k==2*j or k==2*j+1):
                continue
            else:
                else_class = set(np.random.choice(number_list[k], int(i_num * 0.2/8), replace=False))
                for rand in else_class:
                    dict_users[i] = np.concatenate((dict_users[i], [rand]), axis=0)

    return dict_users

'''
def cifar_noniid(dataset_train,dataset_test, num_users):
    num_shards, num_imgs_train,num_imgs_test = 200,300,50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # print(idxs)
    # print(dataset[11924])

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users
'''


def cifar_noniid_2(dataset_train,dataset_test, num_users):
    num_shards, num_imgs_train, num_imgs_test = 200,250,50
    idx_shard = [i for i in range(num_shards)]  # train,test dataset share one idx_shard
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs_train = np.arange(num_shards * num_imgs_train)  # dataset_train index
    idxs_test = np.arange(num_shards * num_imgs_test)  # dataset_test index
    labels_train = np.array(dataset_train.targets)
    labels_test = np.array(dataset_test.targets)
    # print(labels)

    # sort labels
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_train = idxs_labels_train[0, :]
    idxs_test = idxs_labels_test[0, :]


    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs_train[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test[rand * num_imgs_test:(rand + 1) * num_imgs_test]), axis=0)
    return dict_users_train, dict_users_test




if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
