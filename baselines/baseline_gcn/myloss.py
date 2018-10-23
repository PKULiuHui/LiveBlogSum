# coding: utf-8

# 自定义loss函数，与论文中保持一致

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, predict, target):
        loss = -torch.dot(target, torch.log(predict))
        return loss.sum()
