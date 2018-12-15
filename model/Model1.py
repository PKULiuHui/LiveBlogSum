# coding:utf-8

# 层次式encoder，word => sent => doc => live blog
# sent predictor + doc predictor

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class Model1(nn.Module):
    def __init__(self, args, embed=None):
        super(Model1, self).__init__()
        self.model_name = 'Model1'
        self.args = args
        self.V = args.embed_num
        self.D = args.embed_dim
        self.H = args.hidden_size
        self.P = args.pos_dim

        self.embed = nn.Embedding(self.V, self.D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(
            input_size=self.D,
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

        self.doc_RNN = nn.GRU(
            input_size=2 * self.H,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        # position embedding，将sent、doc相对位置映射成一个位置向量
        self.doc_pos_embed = nn.Embedding(self.args.pos_doc_size, self.P)
        self.sent_pos_embed = nn.Embedding(self.args.pos_sent_size, self.P)

        # 预测sent标签时，考虑sent内容，sent与所在doc及blog相关性，sent所在doc的位置，sent在doc中的位置
        self.sent_content = nn.Linear(2 * self.H, 1, bias=False)
        self.sent_salience = nn.Bilinear(2 * self.H, 4 * self.H, 1, bias=False)
        self.sent_doc_pos = nn.Linear(self.P, 1, bias=False)
        self.sent_pos = nn.Linear(self.P, 1, bias=False)
        self.sent_bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

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

    def forward(self, x, doc_lens):  # x: total_sent_num * word_num
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        if self.args.embed_frozen:
            x = self.embed(x).detach()  # total_sent_num * word_num * D
        else:
            x = self.embed(x)
        x, hn = self.word_RNN(x)  # total_sent_num * word_num * (2*H)
        sent_vec = self.max_pool1d(x, sent_lens)  # total_sent_num * (2*H)

        docs = self.seq_split(sent_vec, doc_lens)
        doc_vec = []
        for i, doc in enumerate(docs):
            tmp_h, hn = self.sent_RNN(doc.unsqueeze(0))
            doc_vec.append(self.max_pool1d(tmp_h, [doc_lens[i]]).squeeze(0))
        doc_vec = torch.cat(doc_vec).view(len(doc_lens), -1)  # total_doc_num * (2*H)

        x = doc_vec.unsqueeze(0)  # 1 * total_doc_num * (2*H)
        x, hn = self.doc_RNN(x)  # 1 * total_doc_num * (2*H)
        blog_vec = self.max_pool1d(x, [x.size(1)]).squeeze(0)  # (2*H)
        doc_num = float(len(doc_lens))

        # 预测sent标签
        sent_probs = []
        sent_idx = 0
        for i in range(0, len(doc_lens)):
            context = torch.cat((blog_vec, doc_vec[i]))
            for j in range(0, doc_lens[i]):
                sent_content = self.sent_content(sent_vec[sent_idx])
                sent_salience = self.sent_salience(sent_vec[sent_idx], context)
                sent_doc_index = torch.LongTensor([[int(i * self.args.pos_doc_size / doc_num)]])
                sent_index = torch.LongTensor([[int(j * self.args.pos_sent_size / float(doc_lens[i]))]])
                if use_cuda:
                    sent_doc_index = sent_doc_index.cuda()
                    sent_index = sent_index.cuda()
                sent_doc_pos = self.sent_doc_pos(self.doc_pos_embed(sent_doc_index).squeeze(0))
                sent_pos = self.sent_pos(self.sent_pos_embed(sent_index).squeeze(0))
                sent_pre = sent_content + sent_salience + sent_doc_pos + sent_pos + self.sent_bias
                sent_probs.append(sent_pre)
                sent_idx += 1

        return torch.cat(sent_probs).squeeze()

    @staticmethod
    def seq_split(vecs, seq_lens):
        rst = []
        start = 0
        for seq_len in seq_lens:
            rst.append(vecs[start: start + seq_len])
            start += seq_len
        assert start == len(vecs)
        return rst

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)
