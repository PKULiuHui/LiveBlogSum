# coding: utf-8

# 检验live blog srl预测分数和真是分数的区别

import numpy as np
import torch
import torch.nn.functional as F

path = 'tmp1.txt'
srl_pre = []
srl_tgt = []


def compute(srl_pre, srl_tgt):
    p_5, p_10, p_20 = .0, .0, .0
    mse = .0
    for pre, tgt in zip(srl_pre, srl_tgt):
        mse += F.mse_loss(torch.FloatTensor(pre), torch.FloatTensor(tgt)).data.item()
        idx1 = np.array(pre).argsort().tolist()
        idx1.reverse()
        hit = .0
        for i in idx1[:5]:
            if tgt[i] > 0.00001:
                hit += 1
        p_5 += hit / 5
        hit = .0
        for i in idx1[:10]:
            if tgt[i] > 0.00001:
                hit += 1
        p_10 += hit / 10
        hit = .0
        for i in idx1[:20]:
            if tgt[i] > 0.00001:
                hit += 1
        p_20 += hit / 20
    p_5 /= len(srl_pre)
    p_10 /= len(srl_pre)
    p_20 /= len(srl_pre)
    mse /= len(srl_pre)
    return p_5, p_10, p_20, mse


def main():
    with open(path, 'r') as f:
        for blog in f.read().strip().split('\n\n'):
            cur_pre, cur_tgt = [], []
            for line in blog.strip().split('\n'):
                if len(line.split('\t')) != 2:
                    continue
                try:
                    cur_pre.append(float(line.split('\t')[0]))
                    cur_tgt.append(float(line.split('\t')[1]))
                except:
                    print('error')
                    print(line)
                    exit()
            srl_pre.append(cur_pre)
            srl_tgt.append(cur_tgt)
    p_5, p_10, p_20, mse = compute(srl_pre, srl_tgt)
    print 'P@5:', p_5
    print 'P@10:', p_10
    print 'P@20:', p_20
    print 'mse_loss:', mse


if __name__ == '__main__':
    main()
