from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import pandas as pd

def get_ratio(labels):
    class_weights = np.zeros(max(labels) + 1)
    label_count = Counter(labels)
    for idx, (label, count) in enumerate(sorted(label_count.items(), key=lambda x: -x[1])):
        class_weights[label] = count/len(labels)
    return class_weights

def get_eql_class_weights(lambda_, labels):
    class_weights = np.zeros(max(labels)+1)
    # labels = []
    # with open('datasets/imagenet/annotations/ImageNet_LT_train.txt', 'r') as f:
    #     for lidx, line in enumerate(f):
    #         _, label = line.split()
    #         labels.append(int(label))
    label_count = Counter(labels)
    for idx, (label, count) in enumerate(sorted(label_count.items(), key=lambda x: -x[1])):
        class_weights[label] = 1 if count > lambda_*len(labels) else 0
        #print('idx: {}, cls: {} img: {}, weight: {}'.format(idx, label, count, class_weights[label]))
    return class_weights


def replace_masked_values(tensor, mask, replace_with):
    assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
    one_minus_mask = 1 - mask
    values_to_add = replace_with * one_minus_mask
    return tensor * mask + values_to_add


class SoftmaxEQL(object):
    def __init__(self, labels, lambda_=5e-3, ignore_prob=0.9,): ## based on the default setting of the paper
        self.lambda_ = lambda_
        self.ignore_prob = ignore_prob
        self.class_weight = torch.Tensor(get_eql_class_weights(self.lambda_, labels)).cuda()

    def __call__(self, input, target):
        N, C = input.shape
        not_ignored = self.class_weight.view(1, C).repeat(N, 1)
        over_prob = (torch.rand(input.shape).cuda() > self.ignore_prob).float()
        is_gt = target.new_zeros((N, C)).float()
        is_gt[torch.arange(N), target] = 1

        weights = ((not_ignored + over_prob + is_gt) > 0).float()
        input = replace_masked_values(input, weights, -1e7)
        loss = F.cross_entropy(input, target)
        return loss

class SoftmaxEQL_Action(object):
    def __init__(self, w_v, w_n, index_v, index_n, lambda_=5e-3, ignore_prob=0.9,): ## based on the default setting of the paper
        self.lambda_ = lambda_
        self.ignore_prob = ignore_prob
        self.class_weight = w_v[index_v] * w_n[index_n]

    def __call__(self, input, target):
        N, C = input.shape
        not_ignored = self.class_weight.view(1, C).repeat(N, 1)
        over_prob = (torch.rand(input.shape).cuda() > self.ignore_prob).float()
        is_gt = target.new_zeros((N, C)).float()
        is_gt[torch.arange(N), target] = 1

        weights = ((not_ignored + over_prob + is_gt) > 0).float()
        input = replace_masked_values(input, weights, -1e7)
        loss = F.cross_entropy(input, target)
        return loss


