# coding=utf-8
# coding: utf-8
import sys
import os
import argparse

'''
Standard ROUGE(整个bbc共1803篇blog)
UB1 Rouge-1: 0.480407 Rouge-2: 0.204490 Rouge-l: 0.280785 Rouge-SU4: 0.208131
UB2 Rouge-1: 0.435176 Rouge-2: 0.243138 Rouge-l: 0.280135 Rouge-SU4: 0.209980
LexRank Rouge-1: 0.171248 Rouge-2: 0.030491 Rouge-l: 0.106553 Rouge-SU4: 0.048841
TextRank Rouge-1: 0.145161 Rouge-2: 0.024316 Rouge-l: 0.095294 Rouge-SU4: 0.040450
Luhn Rouge-1: 0.151129 Rouge-2: 0.026597 Rouge-l: 0.097455 Rouge-SU4: 0.042836
ICSI Rouge-1: 0.221558 Rouge-2: 0.055385 Rouge-l: 0.137137 Rouge-SU4: 0.071310
'''

sys.path.append('../../')

from utils.data_helpers import load_data
from tqdm import tqdm
from myrouge.rouge import get_rouge_score

from summarize.upper_bound import ExtractiveUpperbound
from summarize.sume_wrap import SumeWrap
from summarize.sumy.nlp.tokenizers import Tokenizer
from summarize.sumy.parsers.plaintext import PlaintextParser
from summarize.sumy.summarizers.lsa import LsaSummarizer
from summarize.sumy.summarizers.kl import KLSummarizer
from summarize.sumy.summarizers.luhn import LuhnSummarizer
from summarize.sumy.summarizers.lex_rank import LexRankSummarizer
from summarize.sumy.summarizers.text_rank import TextRankSummarizer
from summarize.sumy.nlp.stemmers import Stemmer
from nltk.corpus import stopwords
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
parser = argparse.ArgumentParser(description='LiveBlogSum Baseline')
parser.add_argument('-corpus', type=str, default='bbc_cont_1')
parser.add_argument('-path', type=str, default='../../data/')
parser.add_argument('-sum_len', type=int, default=1)
parser.add_argument('-out', type=str, default='record.txt')

args = parser.parse_args()
args.path = args.path + args.corpus
types = ['train', 'valid', 'test']


def get_summary_scores(algo, docs, refs, summary_size):
    language = 'english'
    summary = ''
    if algo == 'UB1':
        summarizer = ExtractiveUpperbound(language)
        summary = summarizer(docs, refs, summary_size, ngram_type=1)
    elif algo == 'UB2':
        summarizer = ExtractiveUpperbound(language)
        summary = summarizer(docs, refs, summary_size, ngram_type=2)
    elif algo == 'ICSI':
        summarizer = SumeWrap(language)
        summary = summarizer(docs, summary_size)
    else:
        doc_string = u'\n'.join([u'\n'.join(doc_sents) for doc_sents in docs])
        parser = PlaintextParser.from_string(doc_string, Tokenizer(language))
        stemmer = Stemmer(language)
        if algo == 'LSA':
            summarizer = LsaSummarizer(stemmer)
        if algo == 'KL':
            summarizer = KLSummarizer(stemmer)
        if algo == 'Luhn':
            summarizer = LuhnSummarizer(stemmer)
        if algo == 'LexRank':
            summarizer = LexRankSummarizer(stemmer)
        if algo == 'TextRank':
            summarizer = TextRankSummarizer(stemmer)

        summarizer.stop_words = frozenset(stopwords.words(language))
        summary = summarizer(parser.document, summary_size)
    hyps, refs = map(list, zip(*[[' '.join(summary), ' '.join(model)] for model in refs]))
    hyp = str(hyps[0]).split()
    hyp = ' '.join(hyp[:summary_size])
    ref = str(refs[0])
    score = get_rouge_score(hyp, ref)
    return score['ROUGE-1']['r'], score['ROUGE-2']['r'], score['ROUGE-L']['r'], score['ROUGE-SU4']['r']


if __name__ == '__main__':
    out_file = open(args.out, 'w')
    algos = ['UB1', 'UB2', 'LexRank', 'TextRank', 'Luhn', 'ICSI']
    R1 = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    R2 = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    Rl = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    Rsu = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    blog_sum = .0
    for t in types:
        cur_path = args.path + '/' + t + '/'
        file_names = os.listdir(cur_path)
        blog_sum += len(file_names)
        for filename in tqdm(file_names):
            data_file = os.path.join(cur_path, filename)
            docs, refs = load_data(data_file)
            sum_len = len(' '.join(refs[0]).split(' ')) * args.sum_len
            print('####', filename, '####')
            out_file.write(filename + '\n')
            for algo in algos:
                r1, r2, rl, rsu = get_summary_scores(algo, docs, refs, sum_len)
                print algo, r1, r2, rl, rsu
                out_file.write(algo + ' ' + str(r1) + ' ' + str(r2) + ' ' + str(rl) + ' ' + str(rsu) + '\n')
                R1[algo] += r1
                R2[algo] += r2
                Rl[algo] += rl
                Rsu[algo] += rsu
    out_file.close()
    print('Final Results')
    for algo in algos:
        R1[algo] /= blog_sum
        R2[algo] /= blog_sum
        Rl[algo] /= blog_sum
        Rsu[algo] /= blog_sum
        print('%s Rouge-1: %f Rouge-2: %f Rouge-l: %f Rouge-SU4: %f' % (algo, R1[algo], R2[algo], Rl[algo], Rsu[algo]))
