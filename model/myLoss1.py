# coding: utf-8

# 自定义loss函数，sent loss

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class myLoss1(nn.Module):
    def __init__(self):
        super(myLoss1, self).__init__()

    def forward(self, sent_probs, sent_targets, method=1):
        if method == 1:
            return F.mse_loss(sent_probs, sent_targets)
        else:
            zero = torch.FloatTensor([0]).squeeze(0)
            if use_cuda:
                zero = zero.cuda()
            loss = zero
            """
            _, idx = torch.sort(sent_targets)
            for i in range(1, sent_targets.size(0)):
                for j in range(0, i):
                    loss += F.relu(1 - sent_probs[idx[i]] + sent_probs[idx[j]])
            """
            for i in range(0, sent_targets.size(0)):
                for j in range(0, sent_targets.size(0)):
                    if sent_targets[i] > sent_targets[j]:
                        loss += F.relu(1 - sent_probs[i] + sent_probs[j])
            return loss


