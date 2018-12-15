# coding: utf-8

# 自定义loss函数，hinge_loss rank形式

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

use_cuda = torch.cuda.is_available()


class hinge_loss_1(nn.Module):
    def __init__(self, sample=100):
        super(hinge_loss_1, self).__init__()
        self.sample = sample

    def forward(self, probs, targets):
        sample = range(0, len(probs))
        random.shuffle(sample)
        sample = sample[:min(len(probs), self.sample)]
        zero = torch.FloatTensor([0]).squeeze(0)
        if use_cuda:
            zero = zero.cuda()
        loss = zero
        for i in range(0, len(sample)):
            for j in range(0, len(sample)):
                if targets[sample[i]] > targets[sample[j]]:
                    loss += F.relu(1 - probs[sample[i]] + probs[sample[j]])
        return loss
