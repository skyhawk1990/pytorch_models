# -*- coding: utf-8 -*-

from torch import nn, tensor, optim, cat
from torch.utils.data import TensorDataset
from torch.functional import F
import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../')
from data_util import NormalDataReader, DataBunch, Tokenizer, sequence2vocab
from optimization import Learner


##############################################################################################################################################

def text2ids(tokenizer, ds, max_seq_length):
    # 对文本数据进行转换，变为ids
    ids = [tokenizer.convert_tokens_to_ids(d, max_seq_length) for d in ds]
    return ids


def label2ids(label_map, ds):
    # 对标签数据进行转换，变为index
    ids = [label_map[d] for d in ds]
    return ids


class TextCNN(nn.Module):
    # 论文https://arxiv.org/abs/1408.5882
    def __init__(self, vocab_size, embedding_size, kernel_num, kernel_sizes, dropout, label_size):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.convolutions = nn.ModuleList([nn.Conv2d(1, kernel_num, (k, embedding_size)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.batch_normalization = nn.BatchNorm2d(kernel_num)
        self.full_connection = nn.Linear(len(kernel_sizes) * kernel_num, label_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs).unsqueeze(1) # (N, Ci, W, D)
        relu_ouputs = [F.relu(conv(embeddings)) for conv in self.convolutions]  # [(N, Co, W), ...]*len(Ks)
        bn_outputs = [self.batch_normalization(x) for x in relu_ouputs]
        pool_outputs = [F.max_pool1d(x.squeeze(3), x.squeeze(3).size(2)).squeeze(2) for x in bn_outputs]  # [(N, Co), ...]*len(Ks)
        pool_outputs = cat(tuple(pool_outputs), 1)
        pool_outputs = self.dropout(pool_outputs)  # (N, len(Ks)*Co)
        outputs = self.full_connection(pool_outputs)  # (N, C)
        return outputs


def train_and_predict_by_textcnn(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch的lstm和gru模块进行建模.
    测试集准确率分别是
    3层lstm: 0.6545; 3层bi-lstm: 0.6627
    3层gru: 0.6696; 3层bi-gru: 0.6651
    3层gru模型embedding+output作为输出:
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
    model = TextCNN(tokenizer.get_vocab_size(), embedding_size=32, kernel_num=3, kernel_sizes=(3,4,5), dropout=0.3, label_size=len(label_map))
    learner = Learner(data, model)
    learner.fit(epochs=10, lr=0.1, opt_fn=optim.Adagrad)
    learner.accuracy_eval()
    # learner.confusion_eval(label_map)


if __name__ == '__main__':
    train_and_predict_by_textcnn('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
