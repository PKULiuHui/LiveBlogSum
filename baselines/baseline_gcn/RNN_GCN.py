# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class RNN_GCN(nn.Module):
    def __init__(self, args, embed=None):
        super(RNN_GCN, self).__init__()

        self.model_name = 'RNN_GCN'
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.H = args.hidden_size
        self.G = args.gcn_size

        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        self.word_RNN = nn.GRU(
            input_size=D,
            hidden_size=self.H,
            batch_first=True,
        )

        # 每一层都会有不同的权重矩阵，所以组织成列表
        self.graph_w_0 = nn.Parameter(torch.FloatTensor(self.H, self.H).uniform_(-0.1, 0.1))
        self.graph_w_1 = nn.Parameter(torch.FloatTensor(self.H, self.H).uniform_(-0.1, 0.1))
        self.graph_w_2 = nn.Parameter(torch.FloatTensor(self.H, self.H).uniform_(-0.1, 0.1))

        self.sent_RNN = nn.GRU(
            input_size=self.H,
            hidden_size=self.H,
            batch_first=True,
        )

        self.pre_linear_0 = nn.Linear(self.H, self.H, bias=False)
        self.pre_linear_1 = nn.Linear(self.H, self.H, bias=False)
        self.pre_linear_2 = nn.Linear(self.H, 1, bias=False)

        self.content = nn.Linear(self.H, 1, bias=False)
        self.saliance = nn.Bilinear(self.H, self.H, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    # sents: sent_num * sent_trunc, doc_lens: doc_num, sim_matrix: sent_num * sent_num
    def forward(self, sents, doc_lens, sim_matrix):
        # Sentence Embedding
        sents = self.embed(sents)  # sent_num * sent_trunc * D
        _, hn = self.word_RNN(sents)  # hn: 1 * sent_num * H
        sents = hn.squeeze(0)  # sent_num * H

        # GCN
        # sents = F.relu(sim_matrix.mm(sents).mm(self.graph_w_0))  # sent_num * H
        # sents = F.relu(sim_matrix.mm(sents).mm(self.graph_w_1))
        # sents = F.relu(sim_matrix.mm(sents).mm(self.graph_w_2))

        # Doc Embedding
        docs = []
        start = 0
        for doc_len in doc_lens:
            cur_doc = sents[start: start + doc_len]
            cur_doc = cur_doc.unsqueeze(0)
            _, hn = self.sent_RNN(cur_doc)
            docs.append(hn[0][0])
            start += doc_len

        # Blog presentation
        blog = docs[0]
        for doc in docs[1:]:
            blog += doc
        blog /= float(len(docs))

        # predict
        probs = []
        for sent in sents:
            sent_pre = self.pre_linear_2(F.tanh(self.pre_linear_0(blog) + self.pre_linear_1(sent)))
            # sent_pre = self.content(sent) + self.saliance(sent, blog) + self.bias
            probs.append(sent_pre)

        probs = torch.cat(probs).squeeze()
        return F.softmax(probs, dim=0)

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)
