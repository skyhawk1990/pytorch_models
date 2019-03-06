# -*- coding: utf-8 -*-

from torch import nn, tensor, optim
from torch.utils.data import TensorDataset
import numpy as np
import fasttext
import sys
import os
sys.path.append('../')
from data_util import FasttextReader, NormalDataReader, DataBunch, Tokenizer, sequence2vocab
from optimization import Learner


##############################################################################################################################################

def train_and_predict_by_fasttext(train_data_path, valid_data_path):
    """
    利用fasttext包进行训练和预测. 训练集和测试集的准确率分别是0.71和0.68
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    """
    reader = FasttextReader('__label__')
    train_path = '../data/temp/train'
    valid_path = '../data/temp/valid'
    model_path = '../model/fasttext'

    with open(train_path, 'w', encoding='utf-8') as file:
        for v in reader.get_train_data(train_data_path):
            file.write(v + '\n')
    with open(valid_path, 'w', encoding='utf-8') as file:
        for v in reader.get_valid_data(valid_data_path):
            file.write(v + '\n')

    model = fasttext.supervised(train_path, model_path, label_prefix="__label__", epoch=10,
                                word_ngrams=1, min_count=5, bucket=500000, lr=0.1, silent=0, loss='hs')

    fasttext_train_result = model.test(train_path)
    print("FastText acc(train):", fasttext_train_result.precision, fasttext_train_result.recall)
    fasttext_test_result = model.test(valid_path)
    print("FastText acc(test):", fasttext_test_result.precision, fasttext_test_result.recall)


##############################################################################################################################################


def text2ids(tokenizer, ds, max_seq_length):
    # 对文本数据进行转换，变为ids
    ids = [tokenizer.convert_tokens_to_ids(d, max_seq_length) for d in ds]
    return ids


def label2ids(label_map, ds):
    # 对标签数据进行转换，变为index
    ids = [label_map[d] for d in ds]
    return ids


class Lambda(nn.Module):
    # 将函数转换为pytorch中的一层
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


def reduce_mean():
    return Lambda(lambda x: x.mean(dim=1))


def self_fasttext(vocab_size, embedding_size, label_size):
    # 自实现的fasttext
    layers = [nn.Embedding(vocab_size, embedding_size), reduce_mean(), nn.Linear(embedding_size, label_size)]
    return nn.Sequential(*layers)


def train_and_predict_by_self_realization(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch自实现fasttext. 测试集准确率分别是0.713
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    reader = NormalDataReader()
    x_train, y_train = reader.get_train_data(train_data_path)
    x_valid, y_valid = reader.get_valid_data(valid_data_path)
    print('x mean length %s' % (np.mean([len(x) for x in x_train])))

    if not os.path.exists(vocab_path):
        sequence2vocab(x_train, vocab_path)
    tokenizer = Tokenizer(vocab_path, 5)
    print('vocab size %s' % (tokenizer.get_vocab_size()))

    label_map = {label: i for i, label in enumerate(sorted(set(y_train)))}

    x_train = text2ids(tokenizer, x_train, max_seq_length=48)
    y_train = label2ids(label_map, y_train)
    x_valid = text2ids(tokenizer, x_valid, max_seq_length=48)
    y_valid = label2ids(label_map, y_valid)
    print('x_train %s; y_train %s; x_valid %s; y_valid %s' % (len(x_train), len(y_train), len(x_valid), len(y_valid)))
    x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid))
    print(y_train[0], x_train[0])
    print(y_valid[0], x_valid[0])
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    data = DataBunch.create(train_ds, valid_ds, batch_size=64)
    model = self_fasttext(tokenizer.get_vocab_size(), embedding_size=32, label_size=len(label_map))
    learner = Learner(data, model)
    learner.fit(epochs=10, lr=0.1, opt_fn=optim.Adagrad)
    learner.accuracy_eval()
    learner.confusion_eval(label_map)


if __name__=='__main__':
    # train_and_predict_by_fasttext('../data/baike/train.csv', '../data/baike/valid.csv')
    train_and_predict_by_self_realization('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
