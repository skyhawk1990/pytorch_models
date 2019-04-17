# -*- coding: utf-8 -*-

from torch import nn, optim, cat, add, zeros, tanh, Tensor
from torch.nn.parameter import Parameter
from functools import partial
from data import read_data
import sys
sys.path.append('../')
from optimization import Learner


##############################################################################################################################################

class BiLSTM_CNN(nn.Module):
    # BiLSTM接CNN模型
    def __init__(self, vocab_size, embedding_size, hidden_size, max_seq_length, dropout, label_size):
        super(BiLSTM_CNN, self).__init__()

        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, True, True, 0, True)
        self.output_dense = nn.Linear(embedding_size+hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(hidden_size, label_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.lstm(embeddings)
        states = states.view(-1, self.max_seq_length, 2, self.hidden_size)
        states_fw = states[:, :, 0, :]
        states_bw = states[:, :, 1, :]
        outputs = cat((states_fw, embeddings, states_bw), dim=-1)

        outputs = tanh(self.output_dense(outputs))
        outputs = outputs.max(dim=1)[0]
        outputs = self.dropout(outputs)
        outputs = self.full_connection(outputs)
        return outputs


class RCNN(nn.Module):
    # 论文https://arxiv.org/abs/1706.03762
    def __init__(self, vocab_size, embedding_size, hidden_size, max_seq_length, dropout, label_size):
        super(RCNN, self).__init__()

        self.embedding_previous = Parameter(Tensor(1, embedding_size))
        self.context_left_previous = Parameter(Tensor(1, hidden_size))
        self.weight_ls = Parameter(Tensor(max_seq_length, hidden_size, hidden_size))
        self.weight_sls = Parameter(Tensor(max_seq_length, embedding_size, hidden_size))

        self.embedding_following = Parameter(Tensor(1, embedding_size))
        self.context_right_following = Parameter(Tensor(1, hidden_size))
        self.weight_rs = Parameter(Tensor(max_seq_length, hidden_size, hidden_size))
        self.weight_srs = Parameter(Tensor(max_seq_length, embedding_size, hidden_size))

        self.reset_parameters()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.output_dense = nn.Linear(embedding_size + hidden_size*2, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(embedding_size, label_size)

    def reset_parameters(self):
        nn.init.zeros_(self.embedding_previous)
        nn.init.zeros_(self.context_left_previous)
        nn.init.xavier_uniform_(self.weight_ls)
        nn.init.xavier_uniform_(self.weight_sls)

        nn.init.zeros_(self.embedding_following)
        nn.init.zeros_(self.context_right_following)
        nn.init.xavier_uniform_(self.weight_rs)
        nn.init.xavier_uniform_(self.weight_srs)

    def rnn_layer(self, embeddings):
        """
        input:self.embedded_words:[None,sentence_length,embed_size]
        :return: shape:[None,sentence_length,embed_size + hidden_size*2]
        """
        # 1. get size of embeddings
        batch_size, max_seq_length, embedding_size = embeddings.size()

        # 2. get list of context left
        embedding_previous = cat([self.embedding_previous]*batch_size, dim=0)
        context_left_previous = cat([self.context_left_previous]*batch_size, dim=0)
        context_left_list = []
        for i in range(max_seq_length):
            context_left = add(context_left_previous.matmul(self.weight_ls[i, :, :]),
                               embedding_previous.matmul(self.weight_sls[i, :, :]))
            context_left = tanh(context_left)
            context_left_list.append(context_left.unsqueeze(0))
            context_left_previous = context_left
            embedding_previous = embeddings[:, i, :]
        context_left_list = cat(context_left_list, dim=0).permute(1, 0, 2)

        # 3. get list of context right
        embedding_following = cat([self.embedding_following]*batch_size, dim=0)
        context_right_following = cat([self.context_right_following]*batch_size, dim=0)
        context_right_list = []
        for i in range(max_seq_length):
            context_right = add(context_right_following.matmul(self.weight_rs[i, :, :]),
                                embedding_following.matmul(self.weight_srs[i, :, :]))
            context_right = tanh(context_right)
            context_right_list.insert(0, context_right.unsqueeze(0))
            context_right_following = context_right
            embedding_following = embeddings[:, -i - 1, :]
        context_right_list = cat(context_right_list, dim=0).permute(1, 0, 2)

        # 4. ensemble left,embedding,right to output
        output = cat((context_left_list, embeddings, context_right_list), dim=-1)
        return output

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs = self.rnn_layer(embeddings)
        outputs = outputs.max(dim=1)[0]
        outputs = tanh(self.output_dense(outputs))
        outputs = self.dropout(outputs)
        outputs = self.full_connection(outputs)
        return outputs


def train_and_predict_by_rcnn(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch实现RCNN进行建模，并与BiLSTM-CNN对比。
    BiLSTM-CNN: 8轮, 准确率为0.688
    RCNN: 12轮, 准确率为0.659
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    tokenizer, label_map, data = read_data(train_data_path, valid_data_path, vocab_path)
    # model = BiLSTM_CNN(vocab_size=tokenizer.get_vocab_size(),
    #                    embedding_size=32,
    #                    hidden_size=64,
    #                    max_seq_length=48,
    #                    dropout=0.3,
    #                    label_size=len(label_map))
    model = RCNN(vocab_size=tokenizer.get_vocab_size(),
                 embedding_size=32,
                 hidden_size=64,
                 max_seq_length=48,
                 dropout=0.3,
                 label_size=len(label_map))
    learner = Learner(data, model)
    learner.fit(epochs=50, init_lr=0.001, opt_fn=optim.Adam)


if __name__ == '__main__':
    train_and_predict_by_rcnn('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
