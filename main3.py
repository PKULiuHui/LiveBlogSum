# coding:utf-8

# 分阶段训练，先训练SRL打分部分，将打分结果记录下来，然后再训练句子打分部分，直接使用记录的分数
import argparse
import json
import math
import os
import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import model
import utils
from myrouge.rouge import get_rouge_score

parser = argparse.ArgumentParser(description='LiveBlogSum(step by step)')
# model paras
parser.add_argument('-model', type=str, default='Model4')
parser.add_argument('-embed_frozen', type=bool, default=False)
parser.add_argument('-embed_dim', type=int, default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-hidden_size', type=int, default=256)
parser.add_argument('-pos_dim', type=int, default=50)
parser.add_argument('-pos_doc_size', type=int, default=20)  # doc的相对位置个数
parser.add_argument('-pos_sent_size', type=int, default=20)  # sent的相对位置个数
parser.add_argument('-sum_len', type=int, default=1)
parser.add_argument('-mmr', type=float, default=0.75)
# train paras
parser.add_argument('-save_dir', type=str, default='checkpoints7/')
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-lr_decay', type=float, default=0.5)
parser.add_argument('-max_norm', type=float, default=5.0)
parser.add_argument('-srl_epochs', type=int, default=2)  # 训练SRL打分的轮数
parser.add_argument('-sent_epochs', type=int, default=6)  # 训练句子打分的轮数
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-sent_trunc', type=int, default=25)
parser.add_argument('-valid_every', type=int, default=500)
parser.add_argument('-load_model', type=str, default='')
parser.add_argument('-test', action='store_true')
parser.add_argument('-use_cuda', type=bool, default=False)
# data paras
parser.add_argument('-embedding', type=str, default='word2vec/embedding.npz')
parser.add_argument('-word2id', type=str, default='word2vec/word2id.json')
parser.add_argument('-train_dir', type=str, default='data/bbc_srl_4/train/')
parser.add_argument('-valid_dir', type=str, default='data/bbc_srl_4/test/')
parser.add_argument('-test_dir', type=str, default='data/bbc_srl_4/test/')
parser.add_argument('-ref', type=str, default='outputs/ref/')
parser.add_argument('-hyp', type=str, default='outputs/hyp/')

# set random seed, for repeatability
use_cuda = torch.cuda.is_available()
args = parser.parse_args()
if use_cuda:
    torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
args.use_cuda = use_cuda


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
def mmr(sents, scores, ref_len):
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
def evaluate(net, my_loss, vocab, data_iter, srl_scores, train_next):  # train_next指明接下来是否要继续训练
    net.eval()
    my_loss.eval()
    loss, r1, r2, rl, rsu = .0, .0, .0, .0, .0
    blog_num = float(len(data_iter))
    for i, blog in enumerate(tqdm(data_iter)):
        sents, sent_targets, doc_lens, doc_targets, events, event_targets, event_tfs, event_prs, event_lens, event_sent_lens, sents_content, summary = vocab.make_tensors(
            blog, args)
        event_scores = srl_scores[i]
        if use_cuda:
            sents = sents.cuda()
            sent_targets = sent_targets.cuda()
            events = events.cuda()
            event_scores = event_scores.cuda()
        sent_probs = net(sents, doc_lens, events, event_lens, event_scores, False)
        loss += my_loss(sent_probs, sent_targets).data.item()
        probs = sent_probs.tolist()
        ref = summary.strip()
        ref_len = len(ref.split())
        hyp = mmr(sents_content, probs, ref_len)
        score = get_rouge_score(hyp, ref)
        r1 += score['ROUGE-1']['r']
        r2 += score['ROUGE-2']['r']
        rl += score['ROUGE-L']['r']
        rsu += score['ROUGE-SU4']['r']

    loss = loss / blog_num
    r1 = r1 / blog_num
    r2 = r2 / blog_num
    rl = rl / blog_num
    rsu = rsu / blog_num
    if train_next:  # 接下来要继续训练，将网络设成'train'状态
        net.train()
        my_loss.train()
    return loss, r1, r2, rl, rsu


def evaluate_srl(net, my_loss, vocab, data_iter):
    net.eval()
    my_loss.eval()
    loss = .0
    blog_num = float(len(data_iter))
    for i, blog in enumerate(tqdm(data_iter)):
        sents, sent_targets, doc_lens, doc_targets, events, event_targets, event_tfs, event_prs, event_lens, event_sent_lens, sents_content, summary = vocab.make_tensors(
            blog, args)
        if use_cuda:
            sents = sents.cuda()
            events = events.cuda()
            event_targets = event_targets.cuda()
            event_tfs = event_tfs.cuda()
        event_probs = net(sents, doc_lens, events, event_lens, event_sent_lens, event_tfs, True)
        loss += my_loss(event_probs, event_targets).data.item()
    loss = loss / blog_num
    net.train()
    return loss


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (args.lr_decay ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def srl_predict(net, my_loss, vocab, train_data, valid_data):
    print('Begin to predict and record SRL scores...')
    net.eval()
    my_loss.eval()
    train_loss, valid_loss = .0, .0
    train_score, valid_score = [], []
    train_blog_num, valid_blog_num = float(len(train_data)), float(len(valid_data))

    for i, blog in enumerate(tqdm(train_data)):
        sents, sent_targets, doc_lens, doc_targets, events, event_targets, event_tfs, event_prs, event_lens, event_sent_lens, sents_content, summary = vocab.make_tensors(
            blog, args)
        if use_cuda:
            sents = sents.cuda()
            events = events.cuda()
            event_targets = event_targets.cuda()
            event_tfs = event_tfs.cuda()
        event_probs = net(sents, doc_lens, events, event_lens, event_tfs, True)
        train_loss += my_loss(event_probs, event_targets).data.item()
        train_score.append(event_probs.detach())
    train_loss /= train_blog_num

    for i, blog in enumerate(tqdm(valid_data)):
        sents, sent_targets, doc_lens, doc_targets, events, event_targets, event_tfs, event_prs, event_lens, event_sent_lens, sents_content, summary = vocab.make_tensors(
            blog, args)
        if use_cuda:
            sents = sents.cuda()
            events = events.cuda()
            event_targets = event_targets.cuda()
            event_tfs = event_tfs.cuda()
        event_probs = net(sents, doc_lens, events, event_lens, event_tfs, True)
        valid_loss += my_loss(event_probs, event_targets).data.item()
        valid_score.append(event_probs.detach())
    valid_loss /= valid_blog_num

    net.train()
    return train_score, valid_score, train_loss, valid_loss


def train():
    print('Loading vocab, train and valid dataset...')
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    train_data = []
    fns = os.listdir(args.train_dir)
    fns.sort()
    for fn in tqdm(fns):
        f = open(args.train_dir + fn, 'r')
        train_data.append(json.load(f))
        f.close()

    val_data = []
    fns = os.listdir(args.valid_dir)
    fns.sort()
    for fn in tqdm(fns):
        f = open(args.valid_dir + fn, 'r')
        val_data.append(json.load(f))
        f.close()

    net = getattr(model, args.model)(args, embed)
    myloss = nn.MSELoss()
    if use_cuda:
        net.cuda()
        myloss.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()

    # 训练SRL打分
    print('Begin train SRL predictor...')
    for epoch in range(1, args.srl_epochs + 1):
        for i, blog in enumerate(train_data):
            sents, sent_targets, doc_lens, doc_targets, events, event_targets, event_tfs, event_prs, event_lens, event_sent_lens, _1, _2, = vocab.make_tensors(
                blog, args)
            if use_cuda:
                sents = sents.cuda()
                events = events.cuda()
                event_targets = event_targets.cuda()
                event_tfs = event_tfs.cuda()
            event_probs = net(sents, doc_lens, events, event_lens, event_tfs, True)
            loss = myloss(event_probs, event_targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optimizer.step()
            print('SRL EPOCH [%d/%d]: BATCH_ID=[%d/%d] loss=%f' % (epoch, args.srl_epochs, i, len(train_data), loss))
        adjust_learning_rate(optimizer, epoch)
    train_srl_score, valid_srl_score, loss1, loss2 = srl_predict(net, myloss, vocab, train_data, val_data)
    print('SRL predict loss: train: %f valid: %f' % (loss1, loss2))

    # 训练句子打分
    print('Begin train Sent predictor...')
    adjust_learning_rate(optimizer, 0)
    for epoch in range(1, args.sent_epochs + 1):
        for i, blog in enumerate(train_data):
            sents, sent_targets, doc_lens, doc_targets, events, event_targets, event_tfs, event_prs, event_lens, event_sent_lens, _1, _2, = vocab.make_tensors(
                blog, args)
            event_scores = train_srl_score[i]
            if use_cuda:
                sents = sents.cuda()
                sent_targets = sent_targets.cuda()
                events = events.cuda()
                event_scores = event_scores.cuda()
            sent_probs = net(sents, doc_lens, events, event_lens, event_scores, False)
            loss = myloss(sent_probs, sent_targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optimizer.step()
            print('SENT EPOCH [%d/%d]: BATCH_ID=[%d/%d] loss=%f' % (epoch, args.sent_epochs, i, len(train_data), loss))

            cnt = (epoch - 1) * len(train_data) + i
            if cnt % args.valid_every == 0 and cnt / args.valid_every > 0:
                print('Begin valid... Epoch %d, Batch %d' % (epoch, i))
                cur_loss, r1, r2, rl, rsu = evaluate(net, myloss, vocab, val_data, valid_srl_score, True)
                save_path = args.save_dir + args.model + '_SENT_%d_%.4f_%.4f_%.4f_%.4f_%.4f' % (
                    cnt / args.valid_every, cur_loss, r1, r2, rl, rsu)
                net.save(save_path)
                print('Epoch: %2d Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f Rouge-SU4: %f' % (
                    epoch, cur_loss, r1, r2, rl, rsu))
        adjust_learning_rate(optimizer, epoch)


def test():
    # TODO
    return


if __name__ == '__main__':
    if args.test:
        test()
    else:
        train()
