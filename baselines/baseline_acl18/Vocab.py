# coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np


class Vocab:
    def __init__(self, embed, word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v: k for k, v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'

    def __len__(self):
        return len(self.word2id)

    def i2w(self, idx):
        return self.id2word[idx]

    def w2i(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def make_features(self, blog, args):
        summary = ' '.join(blog["summary"])  # 当前blog的summary
        sents = []  # 当前blog的所有句子，用索引表示
        sents_content = []  # 存储原句子
        opt_extract = blog["opt_sents"]  # 存储最佳抽取结果
        sents_target = blog["gain"]  # 存储句子得分，每个句子都有len(opt_extract)个得分

        for sent in blog["sents"]:
            sents.append(sent)
            sents_content.append(sent)

        # 将每一层的所有分数进行Min-Max归一化
        for i, scores in enumerate(sents_target):
            scores = np.array(scores)
            max_score = scores.max()
            min_score = scores.min()
            sents_target[i] = [(tmp - min_score) / (max_score - min_score) for tmp in scores]

        # 将每个句子的单词数截断到sent_trunc，超过截断，不足补全
        for i, sent in enumerate(sents):
            sent = sent.split()
            cur_sent_len = len(sent)
            if cur_sent_len > args.sent_trunc:
                sent = sent[:args.sent_trunc]
            else:
                sent += (args.sent_trunc - cur_sent_len) * [self.PAD_TOKEN]
            sent = [self.w2i(_) for _ in sent]
            sents[i] = sent
        sents = torch.LongTensor(sents)
        targets = torch.FloatTensor(sents_target)
        targets = F.softmax(targets * args.alpha, dim=1)

        return sents, targets, summary, sents_content, opt_extract
