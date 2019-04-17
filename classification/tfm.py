# -*- coding: utf-8 -*-

from torch import nn, optim, tanh, tensor, pow
import numpy as np
from data import read_data
import sys
sys.path.append('../')
from optimization import Learner


##############################################################################################################################################

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + tanh((tensor(np.sqrt(2 / np.pi)) * (x + 0.044715 * pow(x, 3)))))
        return x * cdf


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 max_seq_length,
                 num_layers,
                 attention_dropout,
                 hidden_dropout,
                 dropout,
                 label_size):
        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.size_per_head = int(hidden_size / num_attention_heads)

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.attention_query_layer = nn.ModuleList([nn.Linear(hidden_size, num_attention_heads * self.size_per_head) for _ in range(num_layers)])
        self.attention_key_layer = nn.ModuleList([nn.Linear(hidden_size, num_attention_heads * self.size_per_head) for _ in range(num_layers)])
        self.attention_value_layer = nn.ModuleList([nn.Linear(hidden_size, num_attention_heads * self.size_per_head) for _ in range(num_layers)])
        self.attention_dropout = nn.ModuleList([nn.Dropout(attention_dropout) for _ in range(num_layers)])

        self.transformer_input_layer = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.transformer_input_dropout = nn.ModuleList([nn.Dropout(hidden_dropout) for _ in range(num_layers)])
        self.transformer_input_batch = nn.ModuleList([nn.BatchNorm1d(max_seq_length) for _ in range(num_layers)])
        self.transformer_intermediate_layer = nn.ModuleList([nn.Linear(hidden_size, intermediate_size) for _ in range(num_layers)])
        self.transformer_output_layer = nn.ModuleList([nn.Linear(intermediate_size, hidden_size) for _ in range(num_layers)])
        self.transformer_output_dropout = nn.ModuleList([nn.Dropout(hidden_dropout) for _ in range(num_layers)])
        self.transformer_output_batch = nn.ModuleList([nn.BatchNorm1d(max_seq_length) for _ in range(num_layers)])

        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.full_connection = nn.Linear(hidden_size, label_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs = embeddings
        for i in range(self.num_layers):
            outputs = self.transformer_layer(i, outputs, self.num_attention_heads, self.size_per_head)
        outputs = outputs.mean(dim=1)
        outputs = tanh(self.output_dense(outputs))
        outputs = self.dropout(outputs)
        outputs = self.full_connection(outputs)
        return outputs

    def transformer_layer(self, i, inputs, num_attention_heads, size_per_head):
        attention_output, attention_prob = self.attention_layer(i, inputs, inputs, num_attention_heads, size_per_head)

        attention_output = self.transformer_input_layer[i](attention_output)
        attention_output = self.transformer_input_dropout[i](attention_output)
        attention_output = self.transformer_input_batch[i](attention_output + inputs)

        # The activation is only applied to the "intermediate" hidden layer.
        intermediate_output = self.transformer_intermediate_layer[i](attention_output)
        intermediate_output = GeLU()(intermediate_output)

        # Down-project back to `hidden_size` then add the residual.
        layer_output = self.transformer_output_layer[i](intermediate_output)
        layer_output = self.transformer_output_dropout[i](layer_output)
        layer_output = self.transformer_output_batch[i](layer_output + attention_output)

        return layer_output

    def attention_layer(self, i, from_tensor, to_tensor, num_attention_heads, size_per_head):
        # from_tensor = [batch_size, from_seq_length, from_hidden_size]
        batch_size, from_seq_length, from_hidden_size = from_tensor.size()
        # from_tensor = [batch_size*from_seq_length, from_hidden_size]
        from_tensor = from_tensor.view(batch_size * from_seq_length, from_hidden_size)

        # query_layer = [batch_size*from_seq_length, num_attention_heads*size_per_head]
        query_layer = self.attention_query_layer[i](from_tensor)
        query_layer = query_layer.view(batch_size, from_seq_length, num_attention_heads, size_per_head)
        # query_layer = [batch_size, num_attention_heads, from_seq_length, size_per_head]
        query_layer = query_layer.permute(0, 2, 1, 3)

        # to_tensor = [batch_size, to_seq_length, to_hidden_size]
        batch_size, to_seq_length, to_hidden_size = to_tensor.size()
        # to_tensor = [batch_size*to_seq_length, to_hidden_size]
        to_tensor = to_tensor.view(batch_size * to_seq_length, to_hidden_size)

        # key_layer = [batch_size*to_seq_length, num_attention_heads*size_per_head]
        key_layer = self.attention_key_layer[i](to_tensor)
        key_layer = key_layer.view(batch_size, to_seq_length, num_attention_heads, size_per_head)
        # key_layer = [batch_size, num_attention_heads, to_seq_length, size_per_head]
        key_layer = key_layer.permute(0, 2, 1, 3)

        # value_layer = [batch_size*to_seq_length, num_attention_heads*size_per_head]
        value_layer = self.attention_value_layer[i](to_tensor)
        value_layer = value_layer.view(batch_size, to_seq_length, num_attention_heads, size_per_head)
        # value_layer = [batch_size, num_attention_heads, to_seq_length, size_per_head]
        value_layer = value_layer.permute(0, 2, 1, 3)

        # attention_scores = [batch_size, num_attention_heads, from_seq_length, to_seq_length]
        attention_scores = query_layer.matmul(key_layer.transpose(2, 3))
        attention_scores = attention_scores * 1.0 / np.sqrt(size_per_head)

        # attention_probs = [batch_size, num_attention_heads, from_seq_length, to_seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout[i](attention_probs)

        # context_layer = [batch_size, num_attention_heads, from_seq_length, size_per_head]
        context_layer = attention_probs.matmul(value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)
        # context_layer = [batch_size, from_seq_length, num_attention_heads*size_per_head]
        context_layer = context_layer.reshape(batch_size, from_seq_length, num_attention_heads * size_per_head)

        return context_layer, attention_probs


def train_and_predict_by_transformer(train_data_path, valid_data_path, vocab_path):
    """
    测试transformer模型的效果
    1层, 5轮, 准确率为0.697
    3层, 10轮, 准确率为0.678
    :param train_data_path: 训练集路径
    :param valid_data_path: 测试集路径
    :param vocab_path: 字典路径
    """
    tokenizer, label_map, data = read_data(
        train_data_path, valid_data_path, vocab_path, max_seq_length=48,
        temp_path='../data/temp/train_valid.npy')

    model = Transformer(vocab_size=tokenizer.get_vocab_size(),
                        hidden_size=128,
                        intermediate_size=64,
                        num_attention_heads=4,
                        num_layers=1,
                        max_seq_length=48,
                        attention_dropout=0.1,
                        hidden_dropout=0.1,
                        dropout=0.1,
                        label_size=len(label_map))

    learner = Learner(data, model)
    learner.fit(epochs=20, init_lr=0.001, opt_fn=optim.Adam)
    learner.accuracy_eval()


if __name__ == '__main__':
    train_and_predict_by_transformer('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
