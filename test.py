# coding: utf-8

# 检验live blog中对应预训练embedding的比例，结果有96.7%的词都对应有与训练embedding

import os
import json

word2id_f = './word2vec/word2id.json'
word2id = {}
data_dir = './data/bbc_new/'
types = ['train', 'valid', 'test']


def main():
    with open(word2id_f, 'r') as f:
        word2id = json.load(f)
    print(len(word2id))
    all_cnt = .0
    hit_cnt = .0
    for t in types:
        print(t)
        cur_dir = data_dir + t + '/'
        for fn in os.listdir(cur_dir):
            cur_f = open(cur_dir + fn, 'r')
            blog = json.load(cur_f)
            for doc in blog['documents']:
                for sent in doc['text']:
                    for word in sent.strip().split():
                        all_cnt += 1
                        if word in word2id:
                            hit_cnt += 1
    print(hit_cnt / all_cnt)


if __name__ == '__main__':
    main()
