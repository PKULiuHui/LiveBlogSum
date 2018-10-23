# coding: utf-8

# 有监督学习baseline，为每个句子计算一组特征，从这组特征推断最后句子得分
# 特征包括：surface（和位置相关的特征），content（和内容相关的特征），rel（和句间关系相关的特征）

import json
import re
import math
import os
from tqdm import tqdm
from copy import deepcopy
from nltk.corpus import stopwords
from nltk.text import TextCollection
import numpy as np

train_data = []
valid_data = []
test_data = []
corpus = 'bbc'
train_dir = '../data/' + corpus + '_cont_1/train/'
valid_dir = '../data/' + corpus + '_cont_1/valid/'
test_dir = '../data/' + corpus + '_cont_1/test/'
out_dir = './data/' + corpus + '/'
blog_trunc = 80  # live blog只保留前80个doc
if os.path.exists(out_dir):
    os.system('rm -r ' + out_dir)
os.mkdir(out_dir)
stop_words = stopwords.words('english')


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


def surface(blog, doc_idx, sent_idx, sent):  # 为一个句子计算surface特征
    rst = []
    rst.append(float(doc_idx))  # abs_blog_pos
    rst.append(float(doc_idx) / len(blog.docs))  # rel_blog_pos
    rst.append(float(sent_idx))  # abs_doc_pos
    rst.append(float(sent_idx) / len(blog.docs[doc_idx]))  # rel_doc_pos
    if doc_idx == 0:  # blog_first
        rst.append(1.0)
    else:
        rst.append(.0)
    if sent_idx == 0:  # doc_first
        rst.append(1.0)
    else:
        rst.append(.0)
    rst.append(float(len(sent.split())))  # sent_len
    return rst


def content(blog, doc_idx, sent_idx, sent, text_collection):  # 为一个句子计算content特征，tf, idf, tf_idf
    rst = [.0, .0, .0]  # tf, df, tf_idf的平均值
    cnt = 0
    for w in sent.split():
        if w in stop_words:
            continue
        rst[0] += text_collection.tf(w, blog.docs[doc_idx])
        rst[1] += text_collection.idf(w)
        rst[2] += text_collection.tf_idf(w, blog.docs[doc_idx])
        cnt += 1
    if cnt != 0:
        rst = [t / cnt for t in rst]
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


def PageRank(blog):
    sents = []
    for doc in blog.docs:
        sents.extend(doc)
    sent_num = len(sents)
    unipai = [1.0 / sent_num for i in range(0, sent_num)]
    unipai = np.mat(unipai)  # 平均分布
    pai = deepcopy(unipai)  # 初始得分

    # 根据相似度，计算初始的转移概率矩阵p0
    p0 = []
    for i in range(0, sent_num):
        tmp = [0.0 for i in range(0, sent_num)]
        p0.append(tmp)
    for i in range(0, sent_num):
        for j in range(i + 1, sent_num):
            sim = rouge_1_f(sents[i], sents[j])
            p0[i][j] = p0[j][i] = sim

    # 对转移概率归一化
    for i in range(0, sent_num):
        sumi = sum(p0[i])
        if sumi != 0.0:
            p0[i] = [p0[i][j] / sumi for j in range(0, sent_num)]
    p0 = np.mat(p0)

    iters = 100
    a = 0.85
    for i in range(0, iters):
        oldpai = deepcopy(pai)
        pai = a * oldpai * p0 + (1 - a) * unipai  # pageRank
        # pai几乎不变，则停止迭代
        stop = True
        for j in range(0, sent_num):
            if np.fabs(oldpai[0, j] - pai[0, j]) > 1e-10:
                stop = False
                break
        if stop:
            break
    scores = [pai[0, j] for j in range(0, sent_num)]
    cnt = 0
    rst = deepcopy(blog.scores)
    for i, doc in enumerate(blog.docs):
        for j, sent in enumerate(blog.docs[i]):
            rst[i][j] = scores[cnt]
            cnt += 1
    return rst


def rel(blog, doc_idx, sent_idx, sent):
    rst = [.0, .0]
    if len(blog.docs[0]) > 0:
        rst[0] = rouge_1_f(sent, blog.docs[0][0])
    rst[1] = rouge_1_f(sent, blog.docs[doc_idx][0])
    return rst


def normalize(feats):
    leni = len(feats)
    lenj = len(feats[0])
    Max = [1e-10 for _ in range(0, lenj)]
    Min = [1e10 for _ in range(0, lenj)]
    for feat in feats:
        for j in range(0, lenj):
            Max[j] = max(Max[j], feat[j])
            Min[j] = min(Min[j], feat[j])
    for i in range(0, leni):
        for j in range(0, lenj):
            if Max[j] - Min[j] > 1e-10:
                feats[i][j] = (feats[i][j] - Min[j]) / (Max[j] - Min[j])
            else:
                feats[i][j] = .0
    return feats


def compute_features(blog):
    features = []
    text_collection = []
    for doc in blog.docs:
        text_collection.append(' '.join(doc))
    text_collection = TextCollection(text_collection)  # 为了方便计算tf_idf
    pageranks = PageRank(blog)  # 得到每句话的pagerank分数
    for i, doc in enumerate(blog.docs):
        for j, sent in enumerate(doc):
            cur_feat = []
            cur_feat.extend(surface(blog, i, j, sent))
            cur_feat.extend(content(blog, i, j, sent, text_collection))
            cur_feat.extend(rel(blog, i, j, sent))
            cur_feat.append(pageranks[i][j])
            features.append(cur_feat)
    features = normalize(features)
    return features


def main():
    print('Loading data...')
    for fn in os.listdir(train_dir):
        f = open(os.path.join(train_dir, fn), 'r')
        train_data.append(Blog(json.load(f)))
        f.close()
    for fn in os.listdir(valid_dir):
        f = open(os.path.join(valid_dir, fn), 'r')
        valid_data.append(Blog(json.load(f)))
        f.close()
    for fn in os.listdir(test_dir):
        f = open(os.path.join(test_dir, fn), 'r')
        test_data.append(Blog(json.load(f)))
        f.close()

    print('Computing features...')
    train_features = []
    for blog in tqdm(train_data):
        features = compute_features(blog)
        train_features.extend(features)
    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        for i in range(0, len(train_features)):
            for feat in train_features[i]:
                f.write(str(feat) + ' ')
            f.write('\n')

    valid_features = []
    for blog in tqdm(valid_data):
        features = compute_features(blog)
        valid_features.extend(features)
    with open(os.path.join(out_dir, 'valid.txt'), 'w') as f:
        for i in range(0, len(valid_features)):
            for feat in valid_features[i]:
                f.write(str(feat) + ' ')
            f.write('\n')

    test_features = []
    for blog in tqdm(test_data):
        features = compute_features(blog)
        test_features.extend(features)
    with open(os.path.join(out_dir, 'test.txt'), 'w') as f:
        for i in range(0, len(test_features)):
            for feat in test_features[i]:
                f.write(str(feat) + ' ')
            f.write('\n')


if __name__ == '__main__':
    main()
