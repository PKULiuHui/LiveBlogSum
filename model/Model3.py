# coding:utf-8

# 层次式encoder + SRL attention
# 同时预测sent分数和event分数

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()


class Model3(nn.Module):
    def __init__(self, args, embed=None):
        super(Model3, self).__init__()
        self.model_name = 'Model3'
        self.args = args
        self.V = args.embed_num
        self.D = args.embed_dim
        self.H = args.hidden_size
        self.P = args.pos_dim

        # word embedding层
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

        self.event_RNN = nn.GRU(
            input_size=self.D,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        # position embedding，将sent、doc相对位置映射成一个位置向量
        self.doc_pos_embed = nn.Embedding(self.args.pos_doc_size, self.P)
        self.sent_pos_embed = nn.Embedding(self.args.pos_sent_size, self.P)

        # event预测层，考虑SRL内容，SRL与所在sent、doc、blog的相关性，SRL所在位置，SRL的TF值
        self.event_content = nn.Linear(2 * self.H, 1, bias=False)
        self.event_salience = nn.Bilinear(2 * self.H, 2 * self.H, 1, bias=False)
        self.event_tf = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))
        # self.event_sent = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))
        # self.event_pr = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))
        self.event_bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

        # sent预测层，考虑sent内容，sent与所在doc及blog相关性，sent所在doc的位置，sent在doc中的位置，sent的SRL信息
        self.sent_content = nn.Linear(2 * self.H, 1, bias=False)
        self.sent_salience = nn.Bilinear(2 * self.H, 4 * self.H, 1, bias=False)
        self.sent_doc_pos = nn.Linear(self.P, 1, bias=False)
        self.sent_pos = nn.Linear(self.P, 1, bias=False)
        self.event_rel = nn.Bilinear(2 * self.H, 2 * self.H, 1)  # 计算句子和event相关性
        self.event_para = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))
        self.event_zero = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))
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

    def avg_pool1d(self, x, seq_lens):
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
                out.append(F.avg_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, x, doc_lens, events, event_doc_lens, event_sent_lens, event_tfs, event_targets, sent_targets, train=True):

        doc_num = float(len(doc_lens))

        # SRL(event) presentation
        event_lens = [4] * events.size(0)
        if self.args.embed_frozen:
            events = self.embed(events).detach()
        else:
            events = self.embed(events)
        events, hn = self.event_RNN(events)
        event_vec = self.max_pool1d(events, event_lens)

        t = event_vec.unsqueeze(0)
        event_sum = self.avg_pool1d(t, [t.size(1)]).squeeze(0)
        """
        event_doc_idx = [0] * len(event_vec)  # event位于哪个doc中
        event_sent_abs_idx = [0] * len(event_vec)  # event位于总体的第几句话中
        event_sent_idx = [0] * len(event_vec)  # event位于该doc的第几句话中
        event_doc_sum = self.seq_accumulate(event_doc_lens)
        event_sent_sum = self.seq_accumulate(event_sent_lens)
        event_doc_sent_sum = self.seq_accumulate(doc_lens)
        for i in range(0, len(event_doc_sum)):
            left = 0 if i == 0 else event_doc_sum[i - 1]
            right = event_doc_sum[i]
            for j in range(left, right):
                event_doc_idx[j] = i
        for i in range(0, len(event_sent_sum)):
            left = 0 if i == 0 else event_sent_sum[i - 1]
            right = event_sent_sum[i]
            for j in range(left, right):
                event_sent_idx[j] = i
                event_sent_abs_idx[j] = i
                if event_doc_idx[j] > 0:
                    event_sent_idx[j] -= event_doc_sent_sum[event_doc_idx[j] - 1]
        """
        event_probs = []
        for i, event in enumerate(event_vec):
            event_content = self.event_content(event)
            event_salience = self.event_salience(event, event_sum)
            event_tf = self.event_tf * event_tfs[i]
            # event_sent = self.event_sent * sent_targets[event_sent_abs_idx[i]]
            event_pre = event_content + event_tf + self.event_bias + event_salience
            event_probs.append(event_pre)
        event_probs = torch.cat(event_probs).squeeze()

        """
        # 将event score转为01标签，按比例转
        event_scores = [.0] * len(event_probs)
        idxs = np.argsort(event_probs.detach().cpu().numpy()).tolist()
        idxs.reverse()
        num_one = int(len(event_probs) * self.args.srl_ratio)
        for i in idxs[:num_one]:
            event_scores[i] = 1.0
        event_scores = torch.FloatTensor(event_scores)
        if use_cuda:
            event_scores = event_scores.cuda()
        # 将event_target转为01标签
        event_scores_real = [.0] * len(event_targets)
        idxs = np.argsort(event_targets.detach().cpu().numpy()).tolist()
        idxs.reverse()
        num_one = int(len(event_probs) * self.args.srl_ratio)
        for i in idxs[:num_one]:
            event_scores_real[i] = 1.0
        event_scores_real = torch.FloatTensor(event_scores_real)
        if use_cuda:
            event_scores_real = event_scores_real.cuda()
        """
        """
        # 将event score转成01标签，按阈值转
        event_scores = event_probs.detach().cpu().numpy()
        event_scores = [1 if t > self.args.srl_threshold else 0 for t in event_scores]
        event_scores = torch.FloatTensor(event_scores)
        if use_cuda:
            event_scores = event_scores.cuda()
        # 将event_target转为01标签
        event_scores_real = event_targets.detach().cpu().numpy()
        event_scores_real = [1 if t > self.args.srl_threshold else 0 for t in event_scores_real]
        event_scores_real = torch.FloatTensor(event_scores_real)
        if use_cuda:
            event_scores_real = event_scores_real.cuda()
        """

        # 直接使用event score
        event_scores = event_probs.detach()
        event_scores_real = event_targets

        # word => sent
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        if self.args.embed_frozen:
            x = self.embed(x).detach()  # total_sent_num * word_num * D
        else:
            x = self.embed(x)
        x, hn = self.word_RNN(x)  # total_sent_num * word_num * (2*H)
        sent_vec = self.max_pool1d(x, sent_lens)  # total_sent_num * (2*H)

        # sent => doc
        docs = self.seq_split(sent_vec, doc_lens)
        doc_vec = []
        for i, doc in enumerate(docs):
            tmp_h, hn = self.sent_RNN(doc.unsqueeze(0))
            doc_vec.append(self.max_pool1d(tmp_h, [doc_lens[i]]).squeeze(0))
        doc_vec = torch.cat(doc_vec).view(len(doc_lens), -1)  # total_doc_num * (2*H)

        # doc => blog
        x = doc_vec.unsqueeze(0)  # 1 * total_doc_num * (2*H)
        x, hn = self.doc_RNN(x)  # 1 * total_doc_num * (2*H)
        blog_vec = self.max_pool1d(x, [x.size(1)]).squeeze(0)  # (2*H)

        # sent predictor
        sent_probs = []
        sent_idx = 0
        event_start = 0
        for i in range(0, len(doc_lens)):
            context = torch.cat((blog_vec, doc_vec[i]))
            for j in range(0, doc_lens[i]):
                cur_event_vec = event_vec[event_start: event_start + event_sent_lens[sent_idx]]
                cur_event_prob = event_scores[event_start: event_start + event_sent_lens[sent_idx]]
                cur_event_target = event_scores_real[event_start: event_start + event_sent_lens[sent_idx]]
                event_start += event_sent_lens[sent_idx]
                # 计算当前句子和所在doc的所有event的相似度
                if len(cur_event_vec) == 0:
                    event_rel = self.event_zero
                else:
                    event_sim = self.event_rel(
                        sent_vec[sent_idx].repeat(cur_event_vec.size(0)).view(cur_event_vec.size(0), -1),
                        cur_event_vec).squeeze(1)
                    if train and np.random.random() < self.args.teacher_forcing:
                        event_rel = self.event_para * torch.dot(event_sim, cur_event_target)
                    else:
                        event_rel = self.event_para * torch.dot(event_sim, cur_event_prob)
                cur_sent = sent_vec[sent_idx]
                sent_content = self.sent_content(cur_sent)
                sent_salience = self.sent_salience(cur_sent, context)
                sent_doc_index = torch.LongTensor([[int(i * self.args.pos_doc_size / doc_num)]])
                sent_index = torch.LongTensor([[int(j * self.args.pos_sent_size / doc_lens[i])]])
                if use_cuda:
                    sent_doc_index = sent_doc_index.cuda()
                    sent_index = sent_index.cuda()
                sent_doc_pos = self.sent_doc_pos(self.doc_pos_embed(sent_doc_index).squeeze(0))
                sent_pos = self.sent_pos(self.sent_pos_embed(sent_index).squeeze(0))
                sent_pre = sent_content + sent_salience + sent_doc_pos + sent_pos + event_rel + self.sent_bias
                sent_probs.append(sent_pre)
                sent_idx += 1
        return torch.cat(sent_probs).squeeze(), event_probs

    @staticmethod
    def seq_split(vecs, seq_lens):
        rst = []
        start = 0
        for seq_len in seq_lens:
            rst.append(vecs[start: start + seq_len])
            start += seq_len
        assert start == len(vecs)
        return rst

    @staticmethod
    def seq_accumulate(seq):
        rst = [0] * len(seq)
        for i in range(0, len(seq)):
            if i == 0:
                rst[i] = seq[i]
            else:
                rst[i] = seq[i] + rst[i - 1]
        return rst

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)
