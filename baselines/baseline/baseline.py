# coding:utf-8

# 使用PKUSUMSUM得到summary，并计算rouge值
#
# guardian(L):
# Coverage 0.205367676471 0.0358395294118 0.129378588235 0.0444649705882
# Lead 0.182529264706 0.0286435588235 0.117340205882 0.0370704117647
# Centroid 0.205031264706 0.0322918529412 0.128586235294 0.0436074705882
# LexPageRank 0.213439529412 0.0319259411765 0.130743382353 0.046391
# TextRank 0.228982382353 0.0417126470588 0.142264088235 0.0546559705882
# Submodular 0.253892411765 0.0536839117647 0.154026176471 0.0643656470588
#
# bbc(L):
# Coverage 0.16716185 0.0246575 0.10559635 0.03175085
# Lead 0.1445328 0.0159233 0.09499425 0.0253807
# Centroid 0.16769825 0.02915425 0.10635125 0.03189055
# LexPageRank 0.1832988 0.0290539 0.11394075 0.03591895
# TextRank 0.21151125 0.04204835 0.1329824 0.04681465
# Submodular 0.1995797 0.04061245 0.1227405 0.04277865

import os
import json
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('../')

from tqdm import tqdm
from myrouge.rouge import get_rouge_score

corpus = 'bbc'
# jar_path = '/Users/liuhui/Desktop/Lab/Tools/PKUSUMSUM/PKUSUMSUM.jar'
jar_path = '/home/liuhui/PKUSUMSUM/PKUSUMSUM.jar'
data_dir = '../data/' + corpus + '_cont_1/test/'
tmp_dir = './tmp/'
tmp_out = './out'
sum_len = 1  # 摘要长度是原摘要长度的几倍
methods = [6]
methods_name = {0: 'Coverage', 1: 'Lead', 2: 'Centroid', 3: 'ILP', 4: 'LexPageRank', 5: 'TextRank', 6: 'Submodular'}

if __name__ == '__main__':
    recall = []
    for i in range(0, 7):
        recall.append({'rouge-1': .0, 'rouge-2': .0, 'rouge-l': .0, 'rouge-su*': .0})
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    for fn in tqdm(os.listdir(data_dir)):
        f = open(os.path.join(data_dir, fn), 'r')
        blog = json.load(f)
        f.close()
        ref = str(' '.join(blog['summary']))
        sum_size = len(ref.strip().split()) * sum_len

        if os.path.exists(tmp_dir):  # 保证每次该文件夹都是空的
            for tf in os.listdir(tmp_dir):
                os.remove(tmp_dir + tf)

        for i, doc in enumerate(blog['documents']):
            tmp_f = open(tmp_dir + str(i), 'w')
            for sent in doc['text']:
                tmp_f.write(sent)
                tmp_f.write('\n')
            tmp_f.close()

        for m in methods:
            os.system('java -jar %s -T 2 -input %s -output %s -L 2 -n %d -m %d -stop stopword' % (
                jar_path, tmp_dir, tmp_out, 2 * sum_size, m))
            f = open(tmp_out, 'r')
            hyp = ' '.join(str(f.read()).strip().split()[:sum_size])
            f.close()
            score = get_rouge_score(hyp, ref)
            r1, r2, rl, rsu = score['ROUGE-1']['r'], score['ROUGE-2']['r'], score['ROUGE-L']['r'], score['ROUGE-SU4']['r']
            print methods_name[m], r1, r2, rl, rsu
            recall[m]['rouge-1'] += r1
            recall[m]['rouge-2'] += r2
            recall[m]['rouge-l'] += rl
            recall[m]['rouge-su*'] += rsu

    print('Final Results:')
    for m in methods:
        recall[m]['rouge-1'] /= len(os.listdir(data_dir))
        recall[m]['rouge-2'] /= len(os.listdir(data_dir))
        recall[m]['rouge-l'] /= len(os.listdir(data_dir))
        recall[m]['rouge-su*'] /= len(os.listdir(data_dir))
        print methods_name[m], recall[m]['rouge-1'], recall[m]['rouge-2'], recall[m]['rouge-l'], recall[m]['rouge-su*']
