#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import DatasetSplit


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
            #print(data.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_user(net_user,dataset,idxs,args):
    batch_corrects = []
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
    for batch_id, (data, target) in enumerate(data_loader):
        images, labels = data.to(args.device), target.to(args.device)
        log_probs = net_user(images)
        batch_correct = (torch.sum(torch.argmax(log_probs, dim=1) == labels)).item()
        batch_corrects.append(batch_correct)
    train_acc = sum(batch_corrects) / len(idxs)
    return train_acc
