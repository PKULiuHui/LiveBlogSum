# coding=utf-8
import torch
import torch.nn.functional as F


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
        sim_matrix = torch.FloatTensor(blog["sim_matrix"])
        sents = []  # 当前blog的所有句子，用索引表示
        sents_target = []  # 存储句子得分
        sents_content = []  # 存储原句子
        doc_lens = []  # 存储每个doc所包含的句子数

        for doc in blog["documents"]:
            sents.extend(doc["text"])
            sents_target.extend(doc["sent_label"])
            sents_content.extend(doc["text"])
            doc_lens.append(len(doc["text"]))

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
        targets = F.softmax(targets * args.alpha, dim=0)

        return sents, targets, sim_matrix, doc_lens, sents_content, summary
