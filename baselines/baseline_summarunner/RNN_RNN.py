# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


class RNN_RNN(nn.Module):
    def __init__(self, args, embed=None):
        super(RNN_RNN, self).__init__()
        self.model_name = 'RNN_RNN'
        self.args = args
        V = args.embed_num  # 单词表的大小
        D = args.embed_dim  # 词向量长度
        self.H = args.hidden_size  # 隐藏状态维数
        S = args.seg_num  # 用于计算相对位置，将一篇文章分成固定的块数，句子的块号就是相对位置
        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V, P_D)
        self.rel_pos_embed = nn.Embedding(S, P_D)
        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(
            input_size=D,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        self.sent_RNN = nn.GRU(
            input_size=2 * self.H,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        # 预测sent标签时，考虑sent内容，与blog相关性，冗余性，sent位置，bias
        self.sent_content = nn.Linear(2 * self.H, 1, bias=False)
        self.sent_salience = nn.Bilinear(2 * self.H, 2 * self.H, 1, bias=False)
        self.novelty = nn.Bilinear(2 * self.H, 2 * self.H, 1, bias=False)
        self.abs_pos = nn.Linear(P_D, 1, bias=False)
        self.rel_pos = nn.Linear(P_D, 1, bias=False)
        self.sent_bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

        self.blog_fc = nn.Linear(2 * self.H, 2 * self.H)
        self.sent_fc = nn.Linear(2 * self.H, 2 * self.H)

    def max_pool1d(self, x, seq_lens):
        out = []
        for index, t in enumerate(x):
            if seq_lens[index] == 0:
                if use_cuda:
                    out.append(torch.zeros(1, 2 * self.H, 1).cuda())
                else:
                    out.append(torch.zeros(1, 2 * self.H, 1))
            else:
                t = t[:seq_lens[index], :]
                t = torch.t(t).unsqueeze(0)
                out.append(F.max_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, x, doc_nums, doc_lens):
        # x: total_sent_num * word_num
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        x = self.embed(x)  # total_sent_num * word_num * D
        x = self.word_RNN(x)[0]  # total_sent_num * word_num * (2*H)
        sent_vec = self.max_pool1d(x, sent_lens)  # total_sent_num * (2*H)

        # 现在需要把属于一篇blog的所有句子向量合成blog向量
        blog_lens = []
        doc_lens_start = 0
        for doc_num in doc_nums:
            cur_doc_lens = doc_lens[doc_lens_start: doc_lens_start + doc_num]
            doc_lens_start += doc_num
            blog_lens.append(np.array(cur_doc_lens).sum())

        x = self.padding(sent_vec, blog_lens, self.args.pos_num)  # batch_size * pos_num * (2*H)
        x = self.sent_RNN(x)[0]  # batch_size * pos_num * (2*H)
        blog_vec = self.max_pool1d(x, blog_lens)  # batch_size * (2*H)

        # 预测sent标签
        probs = []
        start = 0
        for i in range(0, len(doc_nums)):
            context = F.tanh(self.blog_fc(blog_vec[i])).view(1,-1)
            end = start + blog_lens[i]
            s = Variable(torch.zeros(1, 2 * self.H))
            if use_cuda:
                s = s.cuda()
            for j in range(start, end):
                sent = F.tanh(self.sent_fc(sent_vec[j])).view(1,-1)
                sent_content = self.sent_content(sent)
                sent_salience = self.sent_salience(sent, context)
                sent_abs_index = torch.LongTensor([[j - start]])
                sent_rel_index = torch.LongTensor([[(j - start) * 10 / blog_lens[i]]])
                if use_cuda:
                    sent_abs_index = sent_abs_index.cuda()
                    sent_rel_index = sent_rel_index.cuda()
                sent_abs_pos = self.abs_pos(self.abs_pos_embed(sent_abs_index).squeeze(0))
                sent_rel_pos = self.rel_pos(self.rel_pos_embed(sent_rel_index).squeeze(0))
                novelty = -1 * self.novelty(sent, s)
                prob = F.sigmoid(sent_content + sent_salience + sent_abs_pos + sent_rel_pos + novelty + self.sent_bias)
                s = s + torch.mm(prob, sent)
                probs.append(prob)
            start = end

        return torch.cat(probs).squeeze()  # 一维tensor，前部分是文档的预测，后部分是所有句子（不含padding）的预测

    # 对于一个序列进行padding，不足的补上全零向量
    def padding(self, vec, seq_lens, trunc):
        pad_dim = vec.size(1)
        result = []
        start = 0
        for seq_len in seq_lens:
            stop = start + seq_len
            valid = vec[start:stop]
            start = stop
            pad = Variable(torch.zeros(trunc - seq_len, pad_dim))
            if use_cuda:
                pad = pad.cuda()
            result.append(torch.cat([valid, pad]).unsqueeze(0))
        result = torch.cat(result, dim=0)
        return result

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)
