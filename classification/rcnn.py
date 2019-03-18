# -*- coding: utf-8 -*-

from torch import nn, optim, cat, zeros, randn, add, tanh
from torch.functional import F
from data import read_data
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../')
from optimization import Learner


##############################################################################################################################################

class BiLSTM_CNN(nn.Module):
    # BiLSTM接CNN模型
    def __init__(self, vocab_size, embedding_size, hidden_size, max_seq_length, kernel_num, kernel_sizes, dropout, label_size):
        super(BiLSTM_CNN, self).__init__()

        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, True, True, 0, True)

        self.convolutions = nn.ModuleList([nn.Conv2d(1, kernel_num, (k, embedding_size + hidden_size * 2)) for k in kernel_sizes])
        self.max_pools = nn.ModuleList([nn.MaxPool2d((max_seq_length - k + 1, 1)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(len(kernel_sizes) * kernel_num, label_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.lstm(embeddings)
        states = states.view(-1, self.max_seq_length, 2, self.hidden_size)
        states_fw = states[:, :, 0, :]
        states_bw = states[:, :, 1, :]
        embeddings = cat((states_fw, embeddings, states_bw), dim=-1).unsqueeze(dim=1)

        relu_ouputs = [F.relu(conv(embeddings)) for conv in self.convolutions]
        pool_outputs = [max_pool(x) for x, max_pool in zip(relu_ouputs, self.max_pools)]
        pool_outputs = cat(pool_outputs, dim=1).squeeze(3).squeeze(2)
        pool_outputs = self.dropout(pool_outputs)
        outputs = self.full_connection(pool_outputs)
        return outputs


class RCNN(nn.Module):
    # 论文https://arxiv.org/abs/1706.03762
    def __init__(self, vocab_size, embedding_size, max_seq_length, dropout, label_size):
        super(RCNN, self).__init__()

        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(embedding_size * 3, label_size)

    def conv_layer(self, embeddings):
        """
        input:self.embedded_words:[None,sentence_length,embed_size]
        :return: shape:[None,sentence_length,embed_size*3]
        """
        # 1. get size of embeddings
        batch_size, max_seq_length, embedding_size = embeddings.size()
        # 2. get list of context left
        embedding_previous = randn((batch_size, embedding_size), requires_grad=True)
        context_left_previous = zeros((batch_size, embedding_size), requires_grad=True)
        context_left_list = []
        for i in range(max_seq_length):
            weight_l = randn((embedding_size, embedding_size), requires_grad=True)
            weight_sl = randn((embedding_size, embedding_size), requires_grad=True)
            context_left = add(context_left_previous.matmul(weight_l), embedding_previous.matmul(weight_sl))
            context_left = tanh(context_left)
            context_left_list.append(context_left.unsqueeze(0))
            context_left_previous = context_left
            embedding_previous = embeddings[:, i, :]
        context_left_list = cat(context_left_list, dim=0)
        # 3. get list of context right
        embedding_following = randn((batch_size, embedding_size), requires_grad=True)
        context_right_following = zeros((batch_size, embedding_size), requires_grad=True)
        context_right_list = []
        for i in range(max_seq_length):
            weight_r = randn((embedding_size, embedding_size), requires_grad=True)
            weight_sr = randn((embedding_size, embedding_size), requires_grad=True)
            context_right = add(context_right_following.matmul(weight_r), embedding_following.matmul(weight_sr))
            context_right = tanh(context_right)
            context_right_list.insert(0, context_right.unsqueeze(0))
            context_right_following = context_right
            embedding_following = embeddings[:, -i - 1, :]
        context_right_list = cat(context_right_list, dim=0)
        # 4. ensemble left,embedding,right to output
        output = cat((context_left_list, embeddings.permute(1, 0, 2), context_right_list), dim=-1)
        output = output.permute(1, 0, 2)
        return output

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs = self.conv_layer(embeddings)
        outputs = outputs.max(dim=1)[0]
        outputs = self.dropout(outputs)
        outputs = self.full_connection(outputs)
        return outputs


def train_and_predict_by_rcnn(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch实现RCNN进行建模，并与BiLSTM-CNN对比。BiLSTM-CNN测试集准确率为0.6692
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    tokenizer, label_map, data = read_data(train_data_path, valid_data_path, vocab_path)
    # model = BiLSTM_CNN(vocab_size=tokenizer.get_vocab_size(),
    #                    embedding_size=32,
    #                    hidden_size=64,
    #                    max_seq_length=48,
    #                    kernel_num=16,
    #                    kernel_sizes=(1, 2, 3),
    #                    dropout=0.1,
    #                    label_size=len(label_map))
    model = RCNN(vocab_size=tokenizer.get_vocab_size(),
                 embedding_size=32,
                 max_seq_length=48,
                 dropout=0.1,
                 label_size=len(label_map))
    learner = Learner(data, model)
    learner.fit(epochs=10, lr=0.1, opt_fn=optim.Adagrad)
    learner.accuracy_eval()
    # learner.confusion_eval(label_map)


if __name__ == '__main__':
    train_and_predict_by_rcnn('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
