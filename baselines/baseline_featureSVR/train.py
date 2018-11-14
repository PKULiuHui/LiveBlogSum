# coding: utf-8

# 利用得到的特征训练SVM回归模型，并预测验证集和测试集的分数

import os
import json
from sklearn.svm import LinearSVR


corpus = 'bbc'
label_method = 'cont_1'
train_label_dir = '../../data/' + corpus + '_' + label_method + '/train/'
feature_dir = './data/' + corpus + '/'


class Blog:
    def __init__(self, blog_json):
        self.id = blog_json['blog_id']
        self.summary = ' '.join(blog_json['summary'])
        self.docs = []
        self.scores = []
        for i, doc in enumerate(blog_json['documents']):
            self.docs.append(doc['text'])
            self.scores.append(doc['sent_label'])


def load_train_label():
    train_data = []
    for fn in os.listdir(train_label_dir):
        f = open(os.path.join(train_label_dir, fn), 'r')
        train_data.append(Blog(json.load(f)))
        f.close()
    train_label = []
    for blog in train_data:
        for score in blog.scores:
            train_label.extend(score)
    return train_label


def Reg(x, y):
    # reg = linear_model.SGDRegressor(max_iter=1000)
    reg = LinearSVR()
    reg.fit(x, y)
    return reg


def main():
    print('Loading data...')
    train_x = []
    with open(feature_dir + 'train.txt', 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            train_x.append([float(_) for _ in data])
    train_y = load_train_label()
    valid_x = []
    with open(feature_dir + 'valid.txt', 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            valid_x.append([float(_) for _ in data])
    test_x = []
    with open(feature_dir + 'test.txt', 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            test_x.append([float(_) for _ in data])

    print('Training model...')
    reg = Reg(train_x, train_y)
    print('Predicting...')
    valid_pre = reg.predict(valid_x)
    test_pre = reg.predict(test_x)
    with open(feature_dir + 'valid_pre.txt', 'w') as f:
        for p in valid_pre:
            f.write(str(p) + '\n')
    with open(feature_dir + 'test_pre.txt', 'w') as f:
        for p in test_pre:
            f.write(str(p) + '\n')


if __name__ == '__main__':
    main()
