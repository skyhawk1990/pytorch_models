# -*- coding: utf-8 -*-

from torch import nn, optim
from data import read_data
import fasttext
import os
import sys
sys.path.append('../')
from data_util import FasttextReader
from optimization import Learner


##############################################################################################################################################

def train_and_predict_by_fasttext(train_data_path, valid_data_path):
    """
    利用fasttext包进行训练和预测. 10轮, 测试集的准确率为0.70
    FAQ: https://fasttext.cc/docs/en/faqs.html
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    """
    reader = FasttextReader('__label__')
    train_path = '../data/temp/train'
    valid_path = '../data/temp/valid'
    model_path = '../model/fasttext'

    if not os.path.exists(train_path):
        with open(train_path, 'w', encoding='utf-8') as file:
            for v in reader.get_train_data(train_data_path):
                file.write(v + '\n')
    if not os.path.exists(valid_path):
        with open(valid_path, 'w', encoding='utf-8') as file:
            for v in reader.get_valid_data(valid_data_path):
                file.write(v + '\n')

    model = fasttext.supervised(train_path, model_path, label_prefix="__label__", epoch=10,
                                word_ngrams=3, min_count=5, bucket=500000, lr=0.1, silent=0, loss='softmax')

    fasttext_train_result = model.test(train_path)
    print("FastText acc(train):", fasttext_train_result.precision, fasttext_train_result.recall)
    fasttext_test_result = model.test(valid_path)
    print("FastText acc(test):", fasttext_test_result.precision, fasttext_test_result.recall)


##############################################################################################################################################


class Lambda(nn.Module):
    # 将函数转换为pytorch中的一层
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def reduce_mean():
    return Lambda(lambda x: x.mean(dim=1))


def self_fasttext(vocab_size, embedding_size, label_size):
    # 自实现的fasttext
    layers = [nn.Embedding(vocab_size, embedding_size), reduce_mean(), nn.Linear(embedding_size, label_size)]
    return nn.Sequential(*layers)


def train_and_predict_by_self_realization(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch自实现fasttext. 14轮达到最优, 测试集准确率是0.723
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    tokenizer, label_map, data = read_data(train_data_path, valid_data_path, vocab_path)
    model = self_fasttext(vocab_size=tokenizer.get_vocab_size(), embedding_size=32, label_size=len(label_map))
    learner = Learner(data, model)
    learner.fit(epochs=20, init_lr=0.001, opt_fn=optim.Adam)


if __name__=='__main__':
    # train_and_predict_by_fasttext('../data/baike/train.csv', '../data/baike/valid.csv')
    train_and_predict_by_self_realization('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
