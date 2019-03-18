# -*- coding: utf-8 -*-

from torch import tensor
from torch.utils.data import TensorDataset
import numpy as np
import sys
import os
sys.path.append('../')
from data_util import NormalDataReader, Tokenizer, sequence2vocab, DataBunch


def text2ids(tokenizer, ds, max_seq_length):
    # 对文本数据进行转换，变为ids
    ids = [tokenizer.convert_tokens_to_ids(d, max_seq_length) for d in ds]
    return ids


def label2ids(label_map, ds):
    # 对标签数据进行转换，变为index
    ids = [label_map[d] for d in ds]
    return ids


def read_data(train_data_path, valid_data_path, vocab_path):
    tokenizer = Tokenizer(vocab_path, 5)
    if os.path.exists('../data/baike/x_train.npy'):
        x_train = np.load('../data/baike/x_train.npy')
        y_train = np.load('../data/baike/y_train.npy')
        x_valid = np.load('../data/baike/x_valid.npy')
        y_valid = np.load('../data/baike/y_valid.npy')
        label_map = {label: i for i, label in enumerate(sorted(set(y_train)))}
    else:
        reader = NormalDataReader()
        x_train, y_train = reader.get_train_data(train_data_path)
        x_valid, y_valid = reader.get_valid_data(valid_data_path)
        print('x mean length %s' % (np.mean([len(x) for x in x_train])))

        if not os.path.exists(vocab_path):
            sequence2vocab(x_train, vocab_path)
        print('vocab size %s' % (tokenizer.get_vocab_size()))

        label_map = {label: i for i, label in enumerate(sorted(set(y_train)))}

        x_train = text2ids(tokenizer, x_train, max_seq_length=48)
        y_train = label2ids(label_map, y_train)
        x_valid = text2ids(tokenizer, x_valid, max_seq_length=48)
        y_valid = label2ids(label_map, y_valid)

        np.save('../data/baike/x_train.npy', x_train)
        np.save('../data/baike/y_train.npy', y_train)
        np.save('../data/baike/x_valid.npy', x_valid)
        np.save('../data/baike/y_valid.npy', y_valid)

    x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid))
    print(y_train[0], x_train[0])
    print(y_valid[0], x_valid[0])
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    data = DataBunch.create(train_ds, valid_ds, batch_size=64)
    return tokenizer, label_map, data
