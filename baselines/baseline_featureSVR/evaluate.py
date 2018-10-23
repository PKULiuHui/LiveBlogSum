# coding: utf-8

# 利用训练得到的分数得到摘要，计算rouge值

import sys

sys.path.append('../')
import json
import re
import os
import math
from myrouge.rouge import get_rouge_score
from tqdm import tqdm

valid_data = []
valid_pre = []
test_data = []
test_pre = []
corpus = 'bbc'
label_method = 'cont_1'
valid_dir = '../data/' + corpus + '_' + label_method + '/valid/'
test_dir = '../data/' + corpus + '_' + label_method + '/test/'
blog_trunc = 80  # live blog只保留前80个doc
pre_dir = './data/' + corpus + '/'
candidate_num = 3  # 得分前15的句子作为候选
mmr = 0.75


class Blog:
    def __init__(self, blog_json):
        self.id = blog_json['blog_id']
        self.summary = ' '.join(blog_json['summary'])
        self.docs = []
        self.scores = []
        for i, doc in enumerate(blog_json['documents']):
            if i >= blog_trunc:
                break
            self.docs.append(doc['text'])
            self.scores.append(doc['sent_label'])


# 根据得分来排序下标
def get_rank(pre):
    a = sorted(enumerate(pre), key=lambda x: x[1])
    rst = []
    for tup in a:
        rst.append(tup[0])
    rst.reverse()
    return rst


# 用rouge_1_f表示两个句子之间的相似度
def rouge_1_f(hyp, ref):
    hyp = re.sub(r'[^a-z]', ' ', hyp.lower()).strip().split()
    ref = re.sub(r'[^a-z]', ' ', ref.lower()).strip().split()
    if len(hyp) == 0 or len(ref) == 0:
        return .0
    ref_flag = [0 for _ in ref]
    hit = .0
    for w in hyp:
        for i in range(0, len(ref)):
            if w == ref[i] and ref_flag[i] == 0:
                hit += 1
                ref_flag[i] = 1
                break
    p = hit / len(hyp)
    r = hit / len(ref)
    if math.fabs(p + r) < 1e-10:
        f = .0
    else:
        f = 2 * p * r / (p + r)
    return f


# 第二种re_rank方法，使用MMR去冗余策略
def re_rank(sents, scores, ref_len):
    sents_num = len(sents)
    sim = [sents_num * [.0] for _ in range(0, sents_num)]
    for i in range(0, sents_num):
        for j in range(i, sents_num):
            if j == i:
                sim[i][j] = 1.0
            else:
                sim[i][j] = sim[j][i] = rouge_1_f(sents[i], sents[j])
    chosen = []
    candidates = range(0, sents_num)
    summary = ''
    cur_len = 0
    while len(candidates) != 0:
        max_point = -1e20
        next = -1
        for i in candidates:
            max_sim = .0
            for j in chosen:
                max_sim = max(max_sim, sim[i][j])
            cur_point = mmr * scores[i] - (1 - mmr) * max_sim
            if cur_point > max_point:
                max_point = cur_point
                next = i
        chosen.append(next)
        candidates.remove(next)
        tmp = sents[next]
        tmp = tmp.split()
        tmp_len = len(tmp)
        if cur_len + tmp_len > ref_len:
            summary += ' '.join(tmp[:ref_len - cur_len])
            break
        else:
            summary += ' '.join(tmp) + ' '
            cur_len += tmp_len
    return summary


def main():
    print('Loading data...')
    for fn in os.listdir(valid_dir):
        f = open(os.path.join(valid_dir, fn), 'r')
        valid_data.append(Blog(json.load(f)))
        f.close()
    with open(pre_dir + 'valid_pre.txt', 'r') as f:
        for line in f.readlines():
            valid_pre.append(float(line))
    for fn in os.listdir(test_dir):
        f = open(os.path.join(test_dir, fn), 'r')
        test_data.append(Blog(json.load(f)))
        f.close()
    with open(pre_dir + 'test_pre.txt', 'r') as f:
        for line in f.readlines():
            test_pre.append(float(line))
    """
    print('Evaluating valid set...')
    r1, r2, rl, rsu = .0, .0, .0, .0
    start = 0
    blog_num = .0
    for blog in tqdm(valid_data):
        sents = []
        for doc in blog.docs:
            sents.extend(doc)
        cur_pre = valid_pre[start: start + len(sents)]
        start = start + len(sents)
        ref_len = len(blog.summary.strip().split())
        hyp = re_rank(sents, cur_pre, ref_len)
        score = get_rouge_score(hyp, blog.summary)
        r1 += score['ROUGE-1']['r']
        r2 += score['ROUGE-2']['r']
        rl += score['ROUGE-L']['r']
        rsu += score['ROUGE-SU4']['r']
        blog_num += 1
    r1 = r1 / blog_num
    r2 = r2 / blog_num
    rl = rl / blog_num
    rsu = rsu / blog_num
    print(r1, r2, rl, rsu)
    """
    print('Evaluating test set...')
    r1, r2, rl, rsu = .0, .0, .0, .0
    start = 0
    blog_num = .0
    for blog in tqdm(test_data):
        sents = []
        for doc in blog.docs:
            sents.extend(doc)
        cur_pre = test_pre[start: start + len(sents)]
        start = start + len(sents)
        ref_len = len(blog.summary.strip().split())
        hyp = re_rank(sents, cur_pre, ref_len)
        score = get_rouge_score(hyp, blog.summary)
        r1 += score['ROUGE-1']['r']
        r2 += score['ROUGE-2']['r']
        rl += score['ROUGE-L']['r']
        rsu += score['ROUGE-SU4']['r']
        blog_num += 1
    r1 = r1 / blog_num
    r2 = r2 / blog_num
    rl = rl / blog_num
    rsu = rsu / blog_num
    print(r1, r2, rl, rsu)


if __name__ == '__main__':
    main()
