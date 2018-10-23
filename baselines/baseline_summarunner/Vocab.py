# coding=utf-8
import torch
import numpy as np
import math


class Vocab():
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

    def make_features(self, batch, args):
        # sent_trunc: 每个句子的词数截取到sent_trunc，不足补全
        # doc_trunc: 每个文档的句子数截取到doc_trunc，不补全
        # blog_trunc: 每个live blog的文档数截取到blog_trunc，不补全
        sent_trunc = args.sent_trunc
        doc_trunc = args.doc_trunc
        blog_trunc = args.blog_trunc

        summarys = []
        for s in batch["summary"]:
            summarys.append(' '.join(s))
        doc_nums = []  # 每个live blog含有多少文档
        for i, d in enumerate(batch["documents"]):
            if len(d) > blog_trunc:
                batch["documents"][i] = d[:blog_trunc]
            doc_nums.append(len(batch["documents"][i]))

        sents = []  # 存储所有句子
        sents_target = []  # 存储所有句子label
        sents_content = []  # 存储所有的句子内容，与sents_target等长，便于之后计算rouge值
        doc_lens = []  # 存储每篇文档包含的句子数
        for d in batch["documents"]:
            for td in d:
                cur_sent_num = len(td["text"])
                if cur_sent_num > doc_trunc:
                    sents.extend(td["text"][:doc_trunc])
                    sents_target.extend(td["sent_label"][:doc_trunc])
                    sents_content.extend(td["text"][:doc_trunc])
                    doc_lens.append(doc_trunc)
                else:
                    sents.extend(td["text"])
                    sents_target.extend(td["sent_label"])
                    sents_content.extend(td["text"])
                    doc_lens.append(cur_sent_num)
        # 将每个句子的单词数固定到sent_trunc，超过截断，不足补全
        for i, sent in enumerate(sents):
            sent = sent.split()
            cur_sent_len = len(sent)
            if cur_sent_len > sent_trunc:
                sent = sent[:sent_trunc]
            else:
                sent += (sent_trunc - cur_sent_len) * [self.PAD_TOKEN]
            sent = [self.w2i(_) for _ in sent]
            sents[i] = sent
        sents = torch.LongTensor(sents)
        targets = sents_target
        targets = torch.FloatTensor(targets)

        return sents, targets, sents_content, summarys, doc_nums, doc_lens
