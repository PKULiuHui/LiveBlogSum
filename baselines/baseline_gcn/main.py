# coding: utf-8

# 实现论文《Graph-based Neural Multi-Document Summarization》中的模型

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from Vocab import Vocab
from RNN_GCN import RNN_GCN
from myloss import myLoss
import numpy as np
import math
import re
import sys
import os, json, argparse, random

sys.path.append('../../')
from myrouge.rouge import get_rouge_score

parser = argparse.ArgumentParser(description='Graph-based Summarization')
parser.add_argument('-save_dir', type=str, default='checkpoints1/')
parser.add_argument('-embed_dim', type=int, default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-hidden_size', type=int, default=256)
parser.add_argument('-gcn_size', type=int, default=3)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-max_norm', type=float, default=1.0)
parser.add_argument('-epochs', type=int, default=8)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-embedding', type=str, default='../../word2vec/embedding.npz')
parser.add_argument('-word2id', type=str, default='../../word2vec/word2id.json')
parser.add_argument('-train_dir', type=str, default='../../data/bbc_graph_1/train/')
parser.add_argument('-valid_dir', type=str, default='../../data/bbc_graph_1/test/')
parser.add_argument('-test_dir', type=str, default='../../data/bbc_graph_1/test/')
parser.add_argument('-sent_trunc', type=int, default=25)
parser.add_argument('-valid_every', type=int, default=500)
parser.add_argument('-load_model', type=str, default='')
parser.add_argument('-greedy', type=float, default=0.3)
parser.add_argument('-alpha', type=float, default=1.0)
parser.add_argument('-sum_len', type=int, default=1)  # 摘要长度为原摘要长度的倍数
parser.add_argument('-test', action='store_true')
parser.add_argument('-use_cuda', type=bool, default=False)

use_cuda = torch.cuda.is_available()
args = parser.parse_args()
if use_cuda:
    torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
args.use_cuda = use_cuda


def train():
    print('Loading vocab, train and val dataset...')
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = Vocab(embed, word2id)
    train_data = []
    for fn in tqdm(os.listdir(args.train_dir)):
        f = open(args.train_dir + fn, 'r')
        train_data.append(json.load(f))
        f.close()
    val_data = []
    for fn in tqdm(os.listdir(args.valid_dir)):
        f = open(args.valid_dir + fn, 'r')
        val_data.append(json.load(f))
        f.close()

    net = RNN_GCN(args, embed)
    criterion = myLoss()
    if use_cuda:
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()
    min_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        for i, blog in enumerate(train_data):
            sents, targets, sim_matrix, doc_lens, _1, _2 = vocab.make_features(blog, args)
            sents, targets, sim_matrix = Variable(sents), Variable(targets), Variable(sim_matrix)
            if use_cuda:
                sents = sents.cuda()
                targets = targets.cuda()
                sim_matrix = sim_matrix.cuda()
            probs = net(sents, doc_lens, sim_matrix)
            loss = criterion(probs, targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optimizer.step()
            print('EPOCH [%d/%d]: BLOG_ID=[%d/%d] loss=%f' % (epoch, args.epochs, i, len(train_data), loss))

            cnt = (epoch - 1) * len(train_data) + i
            if cnt % args.valid_every == 0:
                print('Begin valid... Epoch %d, Batch %d' % (epoch, i))
                cur_loss, r1, r2, rl, rsu = evaluate(net, vocab, val_data, True)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                save_path = args.save_dir + 'RNN_GCN' + '_%d_%.4f_%.4f_%.4f_%.4f_%.4f' % (
                    cnt / args.valid_every, cur_loss, r1, r2, rl, rsu)
                net.save(save_path)
                print('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f Rouge-SU4: %f' %
                      (epoch, min_loss, cur_loss, r1, r2, rl, rsu))
    return


def evaluate(net, vocab, data_iter, train_next):
    net.eval()
    criterion = myLoss()
    loss, r1, r2, rl, rsu = .0, .0, .0, .0, .0
    for blog in tqdm(data_iter):
        sents, targets, sim_matrix, doc_lens, sents_content, summary = vocab.make_features(blog, args)
        sents, targets, sim_matrix = Variable(sents), Variable(targets), Variable(sim_matrix)
        if use_cuda:
            sents = sents.cuda()
            targets = targets.cuda()
            sim_matrix = sim_matrix.cuda()
        probs = net(sents, doc_lens, sim_matrix)
        loss += criterion(probs, targets).data.item()
        hyp = greedy_selection(probs.tolist(), sents_content, len(summary.split()))
        score = get_rouge_score(hyp, summary)
        r1 += score['ROUGE-1']['r']
        r2 += score['ROUGE-2']['r']
        rl += score['ROUGE-L']['r']
        rsu += score['ROUGE-SU4']['r']
    blog_num = len(data_iter)
    loss = loss / blog_num
    r1 = r1 / blog_num
    r2 = r2 / blog_num
    rl = rl / blog_num
    rsu = rsu / blog_num
    if train_next:
        net.train()
    return loss, r1, r2, rl, rsu


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


def greedy_selection(probs, sents_content, ref_len):
    hyp = ""
    selected = []
    sorted_idx = np.array(probs).argsort().tolist()
    sorted_idx.reverse()
    for i in sorted_idx:
        if len(hyp.split()) >= ref_len:
            hyp = " ".join(hyp.split()[:ref_len])
            break
        valid = True
        for j in selected:
            if rouge_1_f(sents_content[i], sents_content[j]) > args.greedy:
                valid = False
                break
        if valid:
            selected.append(i)
            hyp += ' ' + sents_content[i]
    return hyp


def test():
    # TODO
    return


if __name__ == '__main__':
    if args.test:
        test()
    else:
        train()
