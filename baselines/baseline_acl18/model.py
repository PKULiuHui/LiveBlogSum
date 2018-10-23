# coding: utf-8

import torch
import random
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class Model(nn.Module):
    def __init__(self, args, embed=None):
        super(Model, self).__init__()
        self.model_name = 'Model'
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.H = args.hidden_size

        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        # Sentence Embedding层
        self.sent_RNN = nn.GRU(
            input_size=D,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True,
        )
        self.sent_Dropout = nn.Dropout(args.sent_dropout)  # 所以单独增加Dropout层用于处理GRU输出

        # Doc Embedding层
        self.doc_RNN = nn.GRU(
            input_size=2 * self.H,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True,
        )
        self.doc_Dropout = nn.Dropout(args.doc_dropout)

        # 预测层
        self.predict = nn.GRUCell(input_size=2 * self.H, hidden_size=self.H)
        self.wq = nn.Linear(self.H, self.H)
        self.wd = nn.Linear(2 * self.H, self.H)
        self.ws = nn.Linear(self.H, 1)
        self.wm = nn.Linear(self.H, self.H)

    def forward(self, sents, opt):
        # Sentence Embedding，根据Embedding是否保持不变分情况处理
        if self.args.embed_frozen:
            sents = self.embed(sents).detach()
        else:
            sents = self.embed(sents)
        _, hn = self.sent_RNN(sents)
        hn = self.sent_Dropout(hn)
        sents = torch.cat((hn[0], hn[1]), dim=1)  # 1 * sent_num * (2*self.H)
        sents = sents.unsqueeze(0)
        outputs, hn = self.doc_RNN(sents)
        outputs, hn = self.doc_Dropout(outputs), self.doc_Dropout(hn)
        sents = outputs.squeeze(0)  # sent_num * (2*self.H)

        # 预测层
        teacher_forcing = True if random.random() < self.args.teacher_forcing else False
        ht = F.tanh(self.wm(hn[1][0]))  # 初始隐藏状态，由后向GRU最后一个隐藏状态线性变换得到
        st = torch.zeros(2 * self.H)  # 一开始没有选择句子，s0用0向量表示
        if use_cuda:
            st = st.cuda()
        rst = []
        for i in range(0, len(opt)):
            ht = self.predict(st.unsqueeze(0), ht.unsqueeze(0)).squeeze(0)  # ht表示当前已选取的句子集合
            ht1 = ht.repeat(sents.size(0)).view(sents.size(0), -1)
            cur_scores = self.ws(F.tanh(self.wq(ht1) + self.wd(sents))).view(-1)
            cur_scores = F.log_softmax(cur_scores, dim=0)
            rst.append(cur_scores)
            if teacher_forcing:
                st = sents[opt[i]]
            else:
                st = sents[torch.argmax(cur_scores).data.item()]

        return torch.cat(rst).view(len(opt), -1)

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)