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
        srl_trunc = args.srl_trunc
        topic_word_trunc = args.topic_word_trunc

        summarys = []
        for s in batch["summary"]:
            summarys.append(' '.join(s))
        doc_nums = []  # 每个live blog含有多少文档
        doc_targets = []  # 各文档的标签
        for i, d in enumerate(batch["documents"]):
            if len(d) > blog_trunc:
                batch["documents"][i] = d[:blog_trunc]
            doc_nums.append(len(batch["documents"][i]))
            for td in batch["documents"][i]:
                target = td["doc_label"]
                doc_targets.append(target)

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
        targets = doc_targets + sents_target
        targets = torch.FloatTensor(targets)

        events = []  # 存储所有events，即SRL四元组
        event_weights = []  # 存储各events权重
        for d in batch["events"]:
            cur_events = []
            cur_weights = []
            for td in d:
                cur_events.append(td["tuple"])
                cur_weights.append(td["score"])
                if len(cur_events) == srl_trunc:
                    break
            if len(cur_events) < srl_trunc:
                cur_events += (srl_trunc - len(cur_events)) * ["-\t-\t-\t-"]
                cur_weights += (srl_trunc - len(cur_weights)) * [.0]
            cur_weights_sum = np.array(cur_weights).sum()
            cur_weights = [_ / cur_weights_sum for _ in cur_weights]
            events.extend(cur_events)
            event_weights.extend(cur_weights)
        for i, event in enumerate(events):
            event = event.replace('-', self.PAD_TOKEN)
            event = event.strip().split('\t')
            new_event = []
            for w in event:
                if w != self.PAD_TOKEN:
                    new_event.append(w)
            new_event += (4 - len(new_event)) * [self.PAD_TOKEN]
            assert len(new_event) == 4
            event = [self.w2i(_) for _ in new_event]
            events[i] = event
        events = torch.LongTensor(events)
        event_weights = torch.FloatTensor(event_weights)
        return sents, targets, events, event_weights, sents_content, summarys, doc_nums, doc_lens

        topics = []  # 存储所有topics，每个topic存储对应的前几个词
        topic_word_weights = []  # 存储每个word在各topic中的权重
        topic_scores = []  # 存储各个话题的得分 
        for d in batch["topics"]:  # d中存储了一篇blog的所有话题
            for td in d:
                content = td["words"]
                score = td["score"]
                cur_topic = []
                cur_word_weights = []
                for tup in content[0: topic_word_trunc]:
                    cur_topic.append(tup[0])
                    cur_word_weights.append(tup[1])
                cur_word_weight_sum = np.array(cur_word_weights).sum()
                if math.fabs(cur_word_weight_sum) > 1e-5:
                    cur_word_weights = [w / cur_word_weight_sum for w in cur_word_weights]  # 进行归一化
                topics.append(cur_topic)
                topic_word_weights.append(cur_word_weights)
                topic_scores.append(score)
        for i, topic in enumerate(topics):
            topics[i] = [self.w2i(_) for _ in topic]
        topics = torch.LongTensor(topics)
        topic_word_weights = torch.FloatTensor(topic_word_weights)
        topic_scores = torch.FloatTensor(topic_scores)
        return sents, targets, events, event_weights, topics, topic_word_weights, topic_scores, sents_content, summarys, doc_nums, doc_lens
