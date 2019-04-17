# -*- coding: utf-8 -*-

from torch import nn, optim, tanh, randn, matmul
from torch.functional import F
from data import read_data
import sys
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../')
from optimization import Learner
from data_util import to_device


##############################################################################################################################################

def text2ids(tokenizer, ds, max_seq_length):
    # 对文本数据进行转换，变为ids
    num_sentences = 4
    ids = []
    for d in ds:
        temp = []
        for s in re.split('[。，？！,?!]', d):
            if len(s.strip())>0:
                temp.append(tokenizer.convert_tokens_to_ids(s, max_seq_length))
        temp = temp[:num_sentences]
        while len(temp)<num_sentences:
            temp.append([0] * max_seq_length)
        ids.append(temp)
    return ids


class Seq2Seq(nn.Module):
    # 论文：https://arxiv.org/pdf/1409.0473.pdf

    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, label_size):
        super(Seq2Seq, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_encoder = nn.GRU(embedding_size, hidden_size, 1, True, True, 0, True)
        self.word_dense = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.sentence_encoder = nn.GRU(hidden_size * 2, hidden_size * 2, 1, True, True, 0, True)
        self.sentence_dense = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(hidden_size * 4, label_size)

    def forward(self, inputs):
        batch_size, num_sentence, max_seq_length = inputs.size()
        inputs = inputs.view(batch_size * num_sentence, max_seq_length)

        word_emebddings = self.embedding(inputs)
        # word_hidden = [batch_size*num_sentence, max_seq_length*2, hidden_size]
        word_hidden, _ = self.word_encoder(word_emebddings)
        # word_hidden = [batch_size*num_sentence, max_seq_length, hidden_size*2]
        word_hidden = word_hidden.view(batch_size * num_sentence, max_seq_length, -1)

        # sentence_embeddings = [batch_size*num_sentences,hidden_size*2]
        sentence_embeddings = self.word_attention_layer(word_hidden)
        # sentence_embeddings = [batch_size, num_sentence, hidden_size*2]
        sentence_embeddings = sentence_embeddings.view(batch_size, num_sentence, -1)
        # sentence_hidden = [batch_size, num_sentence, hidden_size*4]
        sentence_hidden, _ = self.sentence_encoder(sentence_embeddings)

        # document_embeddings = [batch_size,hidden_size*4]
        document_embeddings = self.sentence_attention_layer(sentence_hidden)
        # document_embeddings = [batch_size,hidden_size*4]
        document_embeddings = self.dropout(document_embeddings)

        outputs = self.full_connection(document_embeddings)
        return outputs

    def word_attention_layer(self, word_hidden):
        # word_representation = [batch_size*num_sentence, max_seq_length, hidden_size*2]
        word_representation = tanh(self.word_dense(word_hidden))
        context_vector_word = to_device(randn((word_representation.size(-1), 1), requires_grad=True))
        # similarity = [batch_size*num_sentences,sequence_length,1]
        similarity = matmul(word_representation, context_vector_word)
        weight = F.softmax(similarity, dim=2)
        # sentence_embeddings = [batch_size*num_sentences,hidden_size*2]
        sentence_embeddings = (weight * word_representation).sum(dim=1, keepdim=False)
        return sentence_embeddings

    def sentence_attention_layer(self, sentence_hidden):
        # sentence_representation = [batch_size, num_sentence, hidden_size*4]
        sentence_representation = tanh(self.sentence_dense(sentence_hidden))
        context_vector_sentence = to_device(randn((sentence_representation.size(-1), 1), requires_grad=True))
        # similarity = [batch_size, num_sentences, 1]
        similarity = matmul(sentence_representation, context_vector_sentence)
        weight = F.softmax(similarity, dim=2)
        # output_embeddings = [batch_size, hidden_size*4]
        output_embeddings = (weight * sentence_representation).sum(dim=1, keepdim=False)
        return output_embeddings


def train_and_predict_by_han(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch的gru模块搭建seq2seq模型。验证集准确率为0.6260
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    tokenizer, label_map, data = read_data(
        train_data_path, valid_data_path, vocab_path, max_seq_length=12,
        temp_path='../data/temp/train_valid.npy', text_func=text2ids)

    model = Seq2Seq(vocab_size=tokenizer.get_vocab_size(),
                                  embedding_size=32,
                                  hidden_size=64,
                                  dropout=0.3,
                                  label_size=len(label_map))

    learner = Learner(data, model)
    learner.fit(epochs=30, lr=0.1, opt_fn=optim.Adagrad)
    learner.accuracy_eval()
    # learner.confusion_eval(label_map)


if __name__ == '__main__':
    train_and_predict_by_han('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
