# coding:utf-8

# 层次式encoder + SRL信息
# 同时预测sent分数和doc分数

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class Model2(nn.Module):
    def __init__(self, args, embed=None):
        super(Model2, self).__init__()
        self.model_name = 'Model2'
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        self.H = args.hidden_size

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

        self.doc_RNN = nn.GRU(
            input_size=2 * self.H,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        # 使用RNN表示events
        self.event_RNN = nn.GRU(
            input_size=D,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        # 预测doc标签时，考虑doc内容，doc与blog相关度，doc相对位置
        self.doc_content = nn.Linear(2 * self.H, 1, bias=False)
        self.doc_salience = nn.Bilinear(2 * self.H, 2 * self.H, 1, bias=False)
        self.doc_pos_embed = nn.Embedding(self.args.pos_doc_size, self.args.pos_dim)
        self.doc_pos = nn.Linear(self.args.pos_dim, 1, bias=False)
        self.doc_bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

        # 预测sent标签时，考虑sent内容，sent与所在doc及blog相关性，sent所在doc的位置，sent在doc中的位置，sent的SRL信息
        self.sent_pre = nn.Linear(4 * self.H, 2 * self.H)
        self.sent_content = nn.Linear(2 * self.H, 1, bias=False)
        self.sent_salience = nn.Bilinear(2 * self.H, 4 * self.H, 1, bias=False)
        self.sent_doc_pos = nn.Linear(self.args.pos_dim, 1, bias=False)
        self.sent_pos_embed = nn.Embedding(self.args.pos_sent_size, self.args.pos_dim)
        self.sent_pos = nn.Linear(self.args.pos_dim, 1, bias=False)
        self.event_zero = nn.Parameter(torch.FloatTensor(2 * self.H).uniform_(-0.1, 0.1))
        self.event_rel = nn.Bilinear(2 * self.H, 2 * self.H, 1)
        # self.event_1 = nn.Linear(2 * self.H, 1, bias=False)
        # self.event_2 = nn.Bilinear(2 * self.H, 4 * self.H, 1, bias=False)
        # self.event_para = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))
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

    def forward(self, x, doc_lens, events, event_doc_lens, event_scores):  # x: total_sent_num * word_num
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        if self.args.embed_frozen:
            x = self.embed(x).detach()  # total_sent_num * word_num * D
        else:
            x = self.embed(x)
        x, hn = self.word_RNN(x)  # total_sent_num * word_num * (2*H)
        sent_vec = self.max_pool1d(x, sent_lens)  # total_sent_num * (2*H)

        docs = self.split(sent_vec, doc_lens)
        doc_vec = []
        for i, doc in enumerate(docs):
            tmp_h, hn = self.sent_RNN(doc.unsqueeze(0))
            doc_vec.append(self.max_pool1d(tmp_h, [doc_lens[i]]).squeeze(0))
        doc_vec = torch.cat(doc_vec).view(len(doc_lens), -1)  # total_doc_num * (2*H)

        x = doc_vec.unsqueeze(0)  # 1 * total_doc_num * (2*H)
        x, hn = self.doc_RNN(x)  # 1 * total_doc_num * (2*H)
        blog_vec = self.max_pool1d(x, [x.size(1)]).squeeze(0)  # (2*H)

        event_lens = [4] * events.size(0)
        if self.args.embed_frozen:
            events = self.embed(events).detach()
        else:
            events = self.embed(events)
        events, hn = self.event_RNN(events)
        event_vec = self.max_pool1d(events, event_lens)

        # 预测doc标签
        doc_probs = []
        doc_num = float(len(doc_lens))
        for i, doc in enumerate(doc_vec):
            doc_content = self.doc_content(doc)
            doc_salience = self.doc_salience(doc, blog_vec)
            doc_index = torch.LongTensor([[int(i * self.args.pos_doc_size / doc_num)]])
            if use_cuda:
                doc_index = doc_index.cuda()
            doc_pos = self.doc_pos(self.doc_pos_embed(doc_index).squeeze(0))
            doc_pre = doc_content + doc_salience + doc_pos + self.doc_bias
            doc_probs.append(doc_pre)

        # 预测sent标签
        sent_probs = []
        sent_idx = 0
        event_start = 0
        for i in range(0, len(doc_lens)):
            context = torch.cat((blog_vec, doc_vec[i]))
            cur_event_vec = event_vec[event_start: event_start + event_doc_lens[i]]
            cur_event_score = event_scores[event_start: event_start + event_doc_lens[i]]
            event_start += event_doc_lens[i]
            for j in range(0, doc_lens[i]):
                if len(cur_event_vec) == 0:
                    # event_rel = self.event_zero
                    event_context = self.event_zero
                else:
                    event_sim = self.event_rel(
                        sent_vec[sent_idx].repeat(cur_event_vec.size(0)).view(cur_event_vec.size(0), -1),
                        cur_event_vec).squeeze(1)
                    event_weight = F.softmax(F.mul(cur_event_score, event_sim), dim=0).unsqueeze(0)
                    event_context = torch.mm(event_weight, cur_event_vec).squeeze(0)
                    # event_rel = self.event_para * torch.dot(event_sim, cur_event_score)
                # event_1 = self.event_1(event_context)
                # event_2 = self.event_2(event_context, context)
                cur_sent = self.sent_pre(torch.cat((sent_vec[sent_idx], event_context)))
                sent_content = self.sent_content(cur_sent)
                sent_salience = self.sent_salience(cur_sent, context)
                sent_doc_index = torch.LongTensor([[int(i * self.args.pos_doc_size / doc_num)]])
                sent_index = torch.LongTensor([[int(j * self.args.pos_sent_size / doc_lens[i])]])
                if use_cuda:
                    sent_doc_index = sent_doc_index.cuda()
                    sent_index = sent_index.cuda()
                sent_doc_pos = self.sent_doc_pos(self.doc_pos_embed(sent_doc_index).squeeze(0))
                sent_pos = self.sent_pos(self.sent_pos_embed(sent_index).squeeze(0))
                sent_pre = sent_content + sent_salience + sent_doc_pos + sent_pos + self.sent_bias
                sent_probs.append(sent_pre)
                sent_idx += 1

        return torch.cat(sent_probs).squeeze(), torch.cat(doc_probs).squeeze()

    def split(self, vecs, seq_lens):
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
