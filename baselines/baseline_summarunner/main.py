# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
import re
import sys
from Vocab import Vocab
from Dataset import Dataset
from RNN_RNN import RNN_RNN
import os, json, argparse, random

sys.path.append('../../')
from myrouge.rouge import get_rouge_score

parser = argparse.ArgumentParser(description='SummaRuNNer')
# model
parser.add_argument('-save_dir', type=str, default='checkpoints1/')
parser.add_argument('-embed_dim', type=int, default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-hidden_size', type=int, default=200)
parser.add_argument('-pos_dim', type=int, default=50)
parser.add_argument('-pos_num', type=int, default=800)
parser.add_argument('-seg_num', type=int, default=10)
# train
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-max_norm', type=float, default=5.0)
parser.add_argument('-batch_size', type=int, default=5)
parser.add_argument('-epochs', type=int, default=8)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-embedding', type=str, default='../../word2vec/embedding.npz')
parser.add_argument('-word2id', type=str, default='../../word2vec/word2id.json')
parser.add_argument('-train_dir', type=str, default='../../data/bbc_opt/train/')
parser.add_argument('-valid_dir', type=str, default='../../data/bbc_opt/test/')
parser.add_argument('-sent_trunc', type=int, default=20)
parser.add_argument('-doc_trunc', type=int, default=10)
parser.add_argument('-blog_trunc', type=int, default=80)
parser.add_argument('-valid_every', type=int, default=100)
# test
parser.add_argument('-load_model', type=str, default='')
parser.add_argument('-test_dir', type=str, default='../../data/bbc_opt/test/')
parser.add_argument('-ref', type=str, default='outputs/ref/')
parser.add_argument('-hyp', type=str, default='outputs/hyp/')
parser.add_argument('-sum_len', type=int, default=1)  # 摘要长度为原摘要长度的倍数
parser.add_argument('-mmr', type=float, default=0.75)
# other
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


def my_collate(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}


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


# 得到预测分数后，使用MMR策略进行重新排序，以消除冗余
def re_rank(sents, scores, ref_len):
    summary = ''
    chosen = []
    cur_scores = [s for s in scores]
    cur_len = 0
    while len(chosen) <= len(scores):
        sorted_idx = np.array(cur_scores).argsort()
        cur_idx = sorted_idx[-1]
        for i in range(len(cur_scores)):
            new_score = args.mmr * scores[i] - (1 - args.mmr) * rouge_1_f(sents[i], sents[cur_idx])
            cur_scores[i] = min(cur_scores[i], new_score)
        cur_scores[cur_idx] = -1e20
        chosen.append(cur_idx)
        tmp = sents[cur_idx].split()
        tmp_len = len(tmp)
        if cur_len + tmp_len > ref_len:
            summary += ' '.join(tmp[:ref_len - cur_len])
            break
        else:
            summary += ' '.join(tmp) + ' '
            cur_len += tmp_len
    return summary.strip()


# 在验证集或测试集上测loss, rouge值
def evaluate(net, vocab, data_iter, train_next):  # train_next指明接下来是否要继续训练
    net.eval()
    criterion = nn.MSELoss()
    loss, r1, r2, rl, rsu = .0, .0, .0, .0, .0  # rouge-1，rouge-2，rouge-l，都使用recall值（长度限定为原摘要长度）
    batch_num = .0
    blog_num = .0
    for i, batch in enumerate(tqdm(data_iter)):
        # 计算loss
        features, targets, sents_content, summaries, doc_nums, doc_lens = vocab.make_features(batch, args)
        features, targets = Variable(features), Variable(targets.float())
        if use_cuda:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_nums, doc_lens)
        batch_num += 1
        loss += criterion(probs, targets).data.item()
        probs_start = 0  # 当前blog对应的probs起始下标
        doc_lens_start = 0  # 当前blog对应的doc_lens起始下标
        sents_start = 0  # 当前blog对应的sents_content起始下标
        for i in range(0, args.batch_size):
            sents_num = 0
            for j in range(doc_lens_start, doc_lens_start + doc_nums[i]):
                sents_num += doc_lens[j]
            cur_probs = probs[probs_start:probs_start + sents_num]
            cur_sents = sents_content[sents_start: sents_start + sents_num]
            probs_start = probs_start + sents_num
            doc_lens_start = doc_lens_start + doc_nums[i]
            sents_start = sents_start + sents_num
            if use_cuda:
                cur_probs = cur_probs.cpu()
            cur_probs = list(cur_probs.detach().numpy())
            sorted_index = list(np.argsort(cur_probs))  # cur_probs顺序排序后对应的下标
            sorted_index.reverse()
            ref = summaries[i].strip()
            ref_len = len(ref.split())
            hyp = re_rank(cur_sents, cur_probs, ref_len)
            score = get_rouge_score(hyp, ref)
            r1 += score['ROUGE-1']['r']
            r2 += score['ROUGE-2']['r']
            rl += score['ROUGE-L']['r']
            rsu += score['ROUGE-SU4']['r']
            blog_num += 1

    loss = loss / batch_num
    r1 = r1 / blog_num
    r2 = r2 / blog_num
    rl = rl / blog_num
    rsu = rsu / blog_num
    if train_next:  # 接下来要继续训练，将网络设成'train'状态
        net.train()
    return loss, r1, r2, rl, rsu


def train():
    print('Loading vocab, train and val dataset...')
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = Vocab(embed, word2id)

    train_data = []
    for fn in os.listdir(args.train_dir):
        f = open(args.train_dir + fn, 'r')
        train_data.append(json.load(f))
        f.close()
    train_dataset = Dataset(train_data)

    val_data = []
    for fn in os.listdir(args.valid_dir):
        f = open(args.valid_dir + fn, 'r')
        val_data.append(json.load(f))
        f.close()
    val_dataset = Dataset(val_data)

    net = RNN_RNN(args, embed)
    criterion = nn.BCELoss()
    if use_cuda:
        net.cuda()

    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=my_collate)

    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          collate_fn=my_collate)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()
    min_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(train_iter):
            features, targets, _1, _2, doc_nums, doc_lens = vocab.make_features(batch, args)
            features, targets = Variable(features), Variable(targets.float())
            if use_cuda:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features, doc_nums, doc_lens)
            loss = criterion(probs, targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optimizer.step()

            print('EPOCH [%d/%d]: BATCH_ID=[%d/%d] loss=%f' % (
                epoch, args.epochs, i, len(train_iter), loss))

            cnt = (epoch - 1) * len(train_iter) + i
            if cnt % args.valid_every == 0:
                print('Begin valid... Epoch %d, Batch %d' % (epoch, i))
                cur_loss, r1, r2, rl, rsu = evaluate(net, vocab, val_iter, True)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                save_path = args.save_dir + 'RNN_RNN' + '_%d_%.4f_%.4f_%.4f_%.4f_%.4f' % (
                    cnt / args.valid_every, cur_loss, r1, r2, rl, rsu)
                net.save(save_path)
                print('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f Rouge-SU4: %f' %
                      (epoch, min_loss, cur_loss, r1, r2, rl, rsu))


def test():
    print('Loading vocab and test dataset...')
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = Vocab(embed, word2id)

    test_data = []
    for fn in os.listdir(args.test_dir):
        f = open(args.test_dir + fn, 'r')
        test_data.append(json.load(f))
        f.close()
    test_dataset = Dataset(test_data)
    test_iter = DataLoader(dataset=test_dataset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           collate_fn=my_collate)
    print('Loading model...')
    if use_cuda:
        checkpoint = torch.load(args.save_dir + args.load_model)
    else:
        checkpoint = torch.load(args.save_dir + args.load_model, map_location=lambda storage, loc: storage)
    net = RNN_RNN(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_cuda:
        net.cuda()
    net.eval()

    print('Begin test...')
    test_loss, r1, r2, rl, rsu = evaluate(net, vocab, test_iter, False)
    print('Test_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f Rouge-SU4: %f' % (test_loss, r1, r2, rl, rsu))


if __name__ == '__main__':
    if args.test:
        test()
    else:
        train()
