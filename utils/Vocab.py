# coding:utf-8
import torch


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

    # Return input and target tensors for training and blog content info for evaluation.
    def make_tensors(self, blog, args):
        summary = ' '.join(blog['summary'])

        doc_targets = []
        for doc in blog['documents']:
            doc_targets.append(doc['doc_label'])

        sents = []
        sents_target = []
        sents_content = []
        doc_lens = []
        for doc in blog['documents']:
            sents.extend(doc['text'])
            sents_target.extend(doc['sent_label'])
            sents_content.extend(doc['text'])
            doc_lens.append(len(doc['text']))
        # 将每个句子的单词数固定到sent_trunc，超过截断，不足补全
        sent_trunc = args.sent_trunc
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
        sents_target = torch.FloatTensor(sents_target)
        doc_targets = torch.FloatTensor(doc_targets)

        events = []  # 存储所有events，即SRL四元组
        event_targets = []  # 存储各events得分，该得分和摘要events计算而得
        event_tfs = []  # 存储各events TF值
        event_lens = []  # 存储每个doc包含的events数目
        event_sent_lens = []  # 存储每个句子包含的events数目

        for doc in blog['documents']:
            cur_len = 0
            for sent_events in doc['events']:
                cur_len += len(sent_events)
                event_sent_lens.append(len(sent_events))
                for event in sent_events:
                    events.append(event['tuple'])
                    event_targets.append(event['score'])
                    event_tfs.append(event['tf'])
            event_lens.append(cur_len)
        for i, event in enumerate(events):
            event = event.replace('-', self.PAD_TOKEN)
            event = event.strip().split('\t')
            event = [self.w2i(_) for _ in event]
            events[i] = event

        events = torch.LongTensor(events)
        event_targets = torch.FloatTensor(event_targets)
        event_tfs = torch.FloatTensor(event_tfs)

        return sents, sents_target, doc_lens, doc_targets, events, event_targets, event_tfs, event_lens, event_sent_lens, sents_content, summary
