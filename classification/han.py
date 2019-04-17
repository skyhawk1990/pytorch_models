# -*- coding: utf-8 -*-

from torch import nn, optim, tanh, matmul, Tensor
from torch.nn.parameter import Parameter
from data import read_data
import sys
sys.path.append('../')
from optimization import Learner


##############################################################################################################################################

# def text2ids(tokenizer, ds, max_seq_length):
#     # 对文本数据进行转换，变为ids
#     num_sentences = 4
#     ids = []
#     for d in ds:
#         temp = []
#         for s in re.split('[。，？！,?!]', d):
#             if len(s.strip()) > 0:
#                 temp.append(tokenizer.convert_tokens_to_ids(s, max_seq_length))
#         temp = temp[:num_sentences]
#         while len(temp) < num_sentences:
#             temp.append([0] * max_seq_length)
#         ids.append(temp)
#     return ids


def text2ids(tokenizer, ds, max_seq_length, num_sentences=12):
    # 对文本数据进行转换，变为ids
    length_per_sentence = max_seq_length//num_sentences
    ids = []
    for d in ds:
        temp = tokenizer.convert_tokens_to_ids(d, max_seq_length)
        ids.append([temp[i*length_per_sentence: (i+1)*length_per_sentence] for i in range(num_sentences)])
    return ids


class SingleLayer(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, label_size):
        super(SingleLayer, self).__init__()

        self.context_vector_word = Parameter(Tensor(hidden_size*2, 1))
        self.reset_parameters()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_encoder = nn.GRU(embedding_size, hidden_size, 1, True, True, 0, True)
        self.word_dense = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.output_dense = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(hidden_size * 2, label_size)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.context_vector_word)

    def forward(self, inputs):
        batch_size, max_seq_length = inputs.size()
        word_emebddings = self.embedding(inputs)
        # word_hidden = [batch_size, max_seq_length*2, hidden_size]
        word_hidden, _ = self.word_encoder(word_emebddings)
        # word_hidden = [batch_size, max_seq_length, hidden_size*2]
        word_hidden = word_hidden.view(batch_size, max_seq_length, -1)

        # output_embeddings = [batch_size, hidden_size*2]
        output_embeddings = self.word_attention_layer(word_hidden)
        # output_embeddings = [batch_size,hidden_size*2]
        # output_embeddings = tanh(self.output_dense(output_embeddings))
        # output_embeddings = self.dropout(output_embeddings)

        outputs = self.full_connection(output_embeddings)
        return outputs

    def word_attention_layer(self, word_hidden):
        # word_representation = [batch_size*num_sentence, max_seq_length, hidden_size*2]
        word_representation = tanh(self.word_dense(word_hidden))

        # similarity = [batch_size*num_sentences,sequence_length,1]
        similarity = matmul(word_representation, self.context_vector_word)
        weight = nn.Softmax(dim=2)(similarity)
        # sentence_embeddings = [batch_size*num_sentences,hidden_size*2]
        output_embeddings = (weight * word_representation).sum(dim=1, keepdim=False)
        return output_embeddings


class HierarchicalAttention(nn.Module):
    # 论文：https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf

    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, label_size):
        super(HierarchicalAttention, self).__init__()

        self.context_vector_word = Parameter(Tensor(hidden_size*2, 1))
        self.context_vector_sentence = Parameter(Tensor(hidden_size*4, 1))
        self.reset_parameters()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_encoder = nn.GRU(embedding_size, hidden_size, 1, True, True, 0, True)
        self.word_dense = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.sentence_encoder = nn.GRU(hidden_size * 2, hidden_size * 2, 1, True, True, 0, True)
        self.sentence_dense = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.document_dense = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(hidden_size * 4, label_size)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.context_vector_word)
        nn.init.xavier_uniform_(self.context_vector_sentence)

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
        # document_embeddings = tanh(self.document_dense(document_embeddings))
        # document_embeddings = self.dropout(document_embeddings)

        outputs = self.full_connection(document_embeddings)
        return outputs

    def word_attention_layer(self, word_hidden):
        # word_representation = [batch_size*num_sentence, max_seq_length, hidden_size*2]
        word_representation = tanh(self.word_dense(word_hidden))
        # similarity = [batch_size*num_sentences,sequence_length,1]
        similarity = matmul(word_representation, self.context_vector_word)
        weight = nn.Softmax(dim=2)(similarity)
        # sentence_embeddings = [batch_size*num_sentences,hidden_size*2]
        sentence_embeddings = (weight * word_representation).sum(dim=1, keepdim=False)
        return sentence_embeddings

    def sentence_attention_layer(self, sentence_hidden):
        # sentence_representation = [batch_size, num_sentence, hidden_size*4]
        sentence_representation = tanh(self.sentence_dense(sentence_hidden))
        # similarity = [batch_size, num_sentences, 1]
        similarity = matmul(sentence_representation, self.context_vector_sentence)
        weight = nn.Softmax(dim=2)(similarity)
        # output_embeddings = [batch_size, hidden_size*4]
        output_embeddings = (weight * sentence_representation).sum(dim=1, keepdim=False)
        return output_embeddings


def train_and_predict_by_han(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch的gru模块搭建han模型。5轮, 准确率为0.678
    尝试只有word层的模型, 9轮, 准确率为0.690
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    # tokenizer, label_map, data = read_data(
    #     train_data_path, valid_data_path, vocab_path, max_seq_length=48,
    #     temp_path='../data/temp/train_valid.npy')
    #
    # model = SingleLayer(vocab_size=tokenizer.get_vocab_size(),
    #                     embedding_size=32,
    #                     hidden_size=64,
    #                     dropout=0.1,
    #                     label_size=len(label_map))
    #
    # learner = Learner(data, model)
    # learner.fit(epochs=50, init_lr=0.001, opt_fn=optim.Adam)

    tokenizer, label_map, data = read_data(
        train_data_path, valid_data_path, vocab_path, max_seq_length=48,
        temp_path='../data/temp/han_train_valid.npy', text_func=text2ids)

    model = HierarchicalAttention(vocab_size=tokenizer.get_vocab_size(),
                                  embedding_size=64,
                                  hidden_size=64,
                                  dropout=0.1,
                                  label_size=len(label_map))

    learner = Learner(data, model)
    learner.fit(epochs=50, init_lr=0.001, opt_fn=optim.Adam)


if __name__ == '__main__':
    train_and_predict_by_han('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
