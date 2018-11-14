# coding: utf-8

# 事先将无监督方法在所有数据上都跑了一遍，记录纸record.txt中，
# 现在划分数据，使得在测试集上的水平接近于整体水平

import random
import os

random.seed(13)
data_dir = '../../data/bbc_cont_1'
types = ['train', 'valid', 'test']
split_num = [0, 1003, 1303, 1803]
blog_names = []


# 加载无监督方法的结果
def load_scores():
    scores = {}
    with open('record.txt', 'r') as f:
        for i in range(0, split_num[-1]):
            cur_fn = f.readline().strip()
            ub1 = [float(_) for _ in f.readline().split()[1:]]
            ub2 = [float(_) for _ in f.readline().split()[1:]]
            lexrank = [float(_) for _ in f.readline().split()[1:]]
            textrank = [float(_) for _ in f.readline().split()[1:]]
            luhn = [float(_) for _ in f.readline().split()[1:]]
            icsi = [float(_) for _ in f.readline().split()[1:]]
            scores[cur_fn] = [ub1, ub2, lexrank, textrank, luhn, icsi]
    return scores


def main():
    scores = load_scores()
    for t in types:
        cur_dir = data_dir + '/' + t + '/'
        fns = os.listdir(cur_dir)
        fns.sort()
        blog_names.extend(fns)
    idx = range(0, split_num[-1])
    # random.shuffle(idx)
    rst = [[.0, .0, .0, .0] for _ in range(0, 6)]
    test_data = [blog_names[i] for i in idx[split_num[-2]: split_num[-1]]]
    for fn in test_data:
        cur_score = scores[fn]
        for i in range(0, 6):
            for j in range(0,4):
                rst[i][j] += cur_score[i][j]
    for i in range(0, 6):
        for j in range(0, 4):
            rst[i][j] /= len(test_data)
    print('UB1', rst[0])
    print('UB2', rst[1])
    print('LexRank', rst[2])
    print('TextRank', rst[3])
    print('Luhn', rst[4])
    print('ICSI', rst[5])


if __name__ == '__main__':
    main()
