# coding: utf-8

# 自定义loss函数，是三种loss（sent, doc, event）的组合

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class myLoss2(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(myLoss2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # self.beta = nn.Parameter(torch.Tensor([beta]))

    def forward(self, sent_probs, doc_probs, event_probs, sent_targets, doc_targets, event_targets):
        loss_1 = F.mse_loss(sent_probs, sent_targets)
        loss_2 = F.mse_loss(doc_probs, doc_targets)
        loss_3 = F.mse_loss(event_probs, event_targets)
        norm = 1.0 + self.alpha + self.beta
        loss = (loss_1 + self.alpha * loss_2 + self.beta * loss_3) / norm
        return loss
