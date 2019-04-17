# -*- coding: utf-8 -*-

from torch import nn, optim, cat
from data import read_data
import sys
sys.path.append('../')
from optimization import Learner


##############################################################################################################################################


class Lambda(nn.Module):
    # 将函数转换为pytorch中的一层
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def reduce_mean():
    return Lambda(lambda x: x[0].mean(dim=1))


def lstm_model(vocab_size, embedding_size, hidden_size, num_layers, bidirectional, dropout, label_size):
    # 自实现的fasttext
    layers = [nn.Embedding(vocab_size, embedding_size),
              nn.LSTM(embedding_size, hidden_size, num_layers, True, True, dropout, bidirectional),
              reduce_mean(),
              nn.Linear(hidden_size*2 if bidirectional else hidden_size, label_size)]
    return nn.Sequential(*layers)


def gru_model(vocab_size, embedding_size, hidden_size, num_layers, dropout, bidirectional, label_size):
    # 自实现的fasttext
    layers = [nn.Embedding(vocab_size, embedding_size),
              nn.GRU(embedding_size, hidden_size, num_layers, True, True, dropout, bidirectional),
              reduce_mean(),
              nn.Linear(hidden_size*2 if bidirectional else hidden_size, label_size)]
    return nn.Sequential(*layers)


class GRUCombineEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, bidirectional, label_size):
        super(GRUCombineEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, True, True, dropout, bidirectional)
        if bidirectional:
            self.decoder = nn.Linear(embedding_size + hidden_size * 2, label_size)
        else:
            self.decoder = nn.Linear(embedding_size + hidden_size, label_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings)
        encoding = cat((embeddings.mean(dim=1), states.mean(dim=1)), dim=1)
        outputs = self.decoder(encoding)
        return outputs


def train_and_predict_by_rnn(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch的lstm和gru模块进行建模. 测试集准确率分别是
    3层lstm: 9轮, 0.68; 3层bi-lstm: 8轮, 0.685
    3层gru: 6轮, 0.684; 3层bi-gru: 8轮, 0.684
    3层gru模型embedding+output作为输出: 9轮, 0.685

    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    tokenizer, label_map, data = read_data(train_data_path, valid_data_path, vocab_path)
    # model = lstm_model(tokenizer.get_vocab_size(),
    #                    embedding_size=32,
    #                    hidden_size=64,
    #                    num_layers=3,
    #                    dropout=0.3,
    #                    bidirectional=True,
    #                    label_size=len(label_map))
    # model = gru_model(tokenizer.get_vocab_size(),
    #                   embedding_size=32,
    #                   hidden_size=64,
    #                   num_layers=3,
    #                   dropout=0.3,
    #                   bidirectional=False,
    #                   label_size=len(label_map))
    model = GRUCombineEmbedding(vocab_size=tokenizer.get_vocab_size(),
                                embedding_size=32,
                                hidden_size=64,
                                num_layers=3,
                                dropout=0.3,
                                bidirectional=True,
                                label_size=len(label_map))
    learner = Learner(data, model)
    learner.fit(epochs=30, init_lr=0.001, opt_fn=optim.Adam)


if __name__ == '__main__':
    train_and_predict_by_rnn('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
