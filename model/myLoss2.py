# coding: utf-8

# 自定义loss函数，是两种loss（sent, doc）的组合

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class myLoss2(nn.Module):
    def __init__(self, alpha=1.0):
        super(myLoss2, self).__init__()
        self.alpha = alpha
        # self.alpha = nn.Parameter(torch.Tensor([alpha]))

    def forward(self, sent_probs, doc_probs, sent_targets, doc_targets):
        loss_1 = F.mse_loss(sent_probs, sent_targets)
        loss_2 = F.mse_loss(doc_probs, doc_targets)
        norm = 1.0 + self.alpha
        loss = (loss_1 + self.alpha * loss_2) / norm
        return loss
