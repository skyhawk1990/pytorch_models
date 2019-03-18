# -*- coding: utf-8 -*-

from torch import nn, optim, cat
from torch.functional import F
from data import read_data
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../')
from optimization import Learner


##############################################################################################################################################


class TextCNN(nn.Module):
    # 论文https://arxiv.org/abs/1408.5882
    def __init__(self, vocab_size, embedding_size, max_seq_length, kernel_num, kernel_sizes, dropout, label_size):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.convolutions = nn.ModuleList([nn.Conv2d(1, kernel_num, (k, embedding_size)) for k in kernel_sizes])
        self.max_pools = nn.ModuleList([nn.MaxPool2d((max_seq_length-k+1, 1)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(len(kernel_sizes) * kernel_num, label_size)

    def forward(self, inputs):
        # embeddings (batch_size, input_size, max_seq_length, embedding_size)
        embeddings = self.embedding(inputs).unsqueeze(1)
        # relu_ouputs [(batch_size, output_size, max_seq_length-kernel_size+1, 1), ...]*len(kernel_sizes)
        relu_ouputs = [F.relu(conv(embeddings)) for conv in self.convolutions]
        # pool_outputs [(batch_size, output_size, 1, 1), ...]*len(kernel_sizes)
        pool_outputs = [max_pool(x) for x, max_pool in zip(relu_ouputs, self.max_pools)]
        # pool_outputs (batch_size, output_size*len(kernel_sizes))
        pool_outputs = cat(pool_outputs, dim=1).squeeze(3).squeeze(2)
        # pool_outputs (batch_size, output_size*len(kernel_sizes))
        pool_outputs = self.dropout(pool_outputs)
        # outputs (batch_size, label_size)
        outputs = self.full_connection(pool_outputs)
        return outputs


def train_and_predict_by_textcnn(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch实现textcnn进行建模. 测试集准确率为0.6865
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    tokenizer, label_map, data = read_data(train_data_path, valid_data_path, vocab_path)
    model = TextCNN(vocab_size=tokenizer.get_vocab_size(),
                    embedding_size=32,
                    max_seq_length=48,
                    kernel_num=32,
                    kernel_sizes=(2,3,4,5),
                    dropout=0.3,
                    label_size=len(label_map))
    learner = Learner(data, model)
    learner.fit(epochs=5, lr=0.1, opt_fn=optim.Adagrad)
    learner.accuracy_eval()
    # learner.confusion_eval(label_map)


if __name__ == '__main__':
    train_and_predict_by_textcnn('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
