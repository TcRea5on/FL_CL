#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_plus(w_pre,w_new,beta):
    w_avg = copy.deepcopy(w_new[0])
    for k in w_avg.keys():
        for i in range(1, len(w_new)):
            w_avg[k] += w_new[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_new))
    return (1 - beta)*w_pre + beta*w_new

