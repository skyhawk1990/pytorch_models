# -*- coding: utf-8 -*-

from torch import nn, tensor, optim, cat
from torch.utils.data import TensorDataset
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

    def attention_word_level(self, hidden_state):
        """
        input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_ = tf.stack(hidden_state, axis=1)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # 0) one layer of feed forward network
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,
                                                          self.hidden_size * 2])  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        # hidden_state_:[batch_size*num_sentences*sequence_length,hidden_size*2];W_w_attention_sentence:[,hidden_size*2,,hidden_size*2]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_word) + self.W_b_attention_word)  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length,
                                                                         self.hidden_size * 2])  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # attention process:1.get logits for each word in the sentence. 2.get possibility distribution for each word in the sentence. 3.get weighted sum for the sentence as sentence representation.
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vecotor_word)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # shape:[batch_size*num_sentences,sequence_length]
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                             keep_dims=True)  # shape:[batch_size*num_sentences,1]
        # 2) get possibility distribution for each word in the sentence.
        p_attention = tf.nn.softmax(
            attention_logits - attention_logits_max)  # shape:[batch_size*num_sentences,sequence_length]
        # 3) get weighted hidden state by attention vector
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[batch_size*num_sentences,sequence_length,1]
        # below sentence_representation'shape:[batch_size*num_sentences,sequence_length,hidden_size*2]<----p_attention_expanded:[batch_size*num_sentences,sequence_length,1];hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation,
                                                axis=1)  # shape:[batch_size*num_sentences,hidden_size*2]
        return sentence_representation  # shape:[batch_size*num_sentences,hidden_size*2]

    def attention_sentence_level(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[None,hidden_size*4]
        input2: sentence level context vector:[self.hidden_size*2]
        :return:representation.shape:[None,hidden_size*4]
        """
        hidden_state_ = tf.stack(hidden_state_sentence, axis=1)  # shape:[None,num_sentence,hidden_size*4]

        # 0) one layer of feed forward
        hidden_state_2 = tf.reshape(hidden_state_,
                                    shape=[-1, self.hidden_size * 4])  # [None*num_sentence,hidden_size*4]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_sentence) + self.W_b_attention_sentence)  # shape:[None*num_sentence,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,
                                                                         self.hidden_size * 2])  # [None,num_sentence,hidden_size*2]
        # attention process:1.get logits for each sentence in the doc.2.get possibility distribution for each sentence in the doc.3.get weighted sum for the sentences as doc representation.
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vecotor_sentence)  # shape:[None,num_sentence,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # shape:[None,num_sentence]. that is get logit for each num_sentence.
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)  # shape:[None,1]
        # 2) get possibility distribution for each word in the sentence.
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)  # shape:[None,num_sentence]
        # 3) get weighted hidden state by attention vector(sentence level)
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[None,num_sentence,1]
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[None,num_sentence,hidden_size*2]<---p_attention_expanded:[None,num_sentence,1];hidden_state_:[None,num_sentence,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)  # shape:[None,hidden_size*2]
        return sentence_representation  # shape:[None,hidden_size*2]

    def inference(self):
        """main computation graph here: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier"""
        # 1.Word Encoder
        # 1.1 embedding of words
        input_x = tf.split(self.input_x, self.num_sentences,
                           axis=1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        input_x = tf.stack(input_x, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,
                                                     input_x)  # [None,num_sentences,sentence_length,embed_size]
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length,
                                                                         self.embed_size])  # [batch_size*num_sentences,sentence_length,embed_size]
        # 1.2 forward gru
        hidden_state_forward_list = self.gru_forward_word_level(
            embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.3 backward gru
        hidden_state_backward_list = self.gru_backward_word_level(
            embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.4 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list,
                                 hidden_state_backward_list)]  # hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]

        # 2.Word Attention
        # for each sentence.
        sentence_representation = self.attention_word_level(
            self.hidden_state)  # output:[batch_size*num_sentences,hidden_size*2]
        sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.num_sentences,
                                                                             self.hidden_size * 2])  # shape:[batch_size,num_sentences,hidden_size*2]
        # with tf.name_scope("dropout"):#TODO
        #    sentence_representation = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]

        # 3.Sentence Encoder
        # 3.1) forward gru for sentence
        hidden_state_forward_sentences = self.gru_forward_sentence_level(
            sentence_representation)  # a list.length is sentence_length, each element is [None,hidden_size]
        # 3.2) backward gru for sentence
        hidden_state_backward_sentences = self.gru_backward_sentence_level(
            sentence_representation)  # a list,length is sentence_length, each element is [None,hidden_size]
        # 3.3) concat forward hidden state and backward hidden state
        # below hidden_state_sentence is a list,len:sentence_length,element:[None,hidden_size*2]
        self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                                      zip(hidden_state_forward_sentences, hidden_state_backward_sentences)]

        # 4.Sentence Attention
        document_representation = self.attention_sentence_level(
            self.hidden_state_sentence)  # shape:[None,hidden_size*4]
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(document_representation,
                                        keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop,
                               self.W_projection) + self.b_projection  # shape:[None,self.num_classes]==tf.matmul([None,hidden_size*2],[hidden_size*2,self.num_classes])
        return logits

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_multilabel(self,
                        l2_lambda=0.00001 * 10):  # *3#0.00001 #TODO 0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

class HierarchicalAttention(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, bidirectional, label_size):
        super(HierarchicalAttention, self).__init__()
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


def train_and_predict_by_han(train_data_path, valid_data_path, vocab_path):
    """
    利用pytorch的lstm和gru模块进行建模. 测试集准确率分别是
    3层lstm: 0.6545; 3层bi-lstm: 0.6627
    3层gru: 0.6696; 3层bi-gru: 0.6651
    3层gru模型embedding+output作为输出: 0.6972

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
    model = HierarchicalAttention(tokenizer.get_vocab_size(),
                                embedding_size=32,
                                hidden_size=64,
                                num_layers=3,
                                dropout=0.3,
                                bidirectional=True,
                                label_size=len(label_map))
    learner = Learner(data, model)
    learner.fit(epochs=5, lr=0.1, opt_fn=optim.Adagrad)
    learner.accuracy_eval()
    # learner.confusion_eval(label_map)


if __name__ == '__main__':
    train_and_predict_by_han('../data/baike/train.csv', '../data/baike/valid.csv', '../model/vocab')
