# -*- coding: utf-8 -*-

import jieba
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


##############################################################################################################################################

class DataReader():

    def get_train_data(self, train_data_path):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_valid_data(self, valid_data_path):
        """Gets a collection of `InputExample`s for the valid set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, data_path):
        """Reads a comma separated value file."""
        data = pd.read_csv(data_path)
        return data


class NormalDataReader(DataReader):

    def get_train_data(self, train_data_path):
        return self._create_data(self._read_csv(train_data_path))

    def get_valid_data(self, valid_data_path):
        return self._create_data(self._read_csv(valid_data_path))

    def _create_data(self, data):
        return data['x'].values, data['y'].values


class FasttextReader(DataReader):

    def __init__(self, split_pattern):
        self.split_pattern = split_pattern

    def get_train_data(self, train_data_path):
        return self._create_data(self._read_csv(train_data_path))

    def get_valid_data(self, valid_data_path):
        return self._create_data(self._read_csv(valid_data_path))

    def _create_data(self, data):
        data['words'] = data['x'].apply(lambda x: ' '.join(jieba.lcut(x)))
        return (data['words'] + ' ' + self.split_pattern + data['y']).values


##############################################################################################################################################

class DatasetTfm(Dataset):
    # 对数据进行转换
    def __init__(self, ds, tfm, *args):
        self.ds = ds
        self.tfm = tfm
        self.args = args

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # 对数据进行转换并返回第idx个元素
        x, y = self.ds[idx]
        if self.tfm is not None:
            x = self.tfm(x, *self.args)
        return x, y


##############################################################################################################################################

default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def ifnone(a, b):
    return b if a is None else a


def is_listy(x):
    return isinstance(x, (tuple, list))


def to_device(b, device=None):
    # 将tensors转换到gpu上
    device = ifnone(device, default_device)
    if is_listy(b):
        return [to_device(o, device) for o in b]
    return b.to(device)


class DeviceDataLoader():
    # 载入数据并保证数据在gpu上
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __len__(self):
        return len(self.dl)

    def proc_batch(self, b):
        return to_device(b, self.device)

    def __iter__(self):
        self.gen = map(self.proc_batch, self.dl)
        return iter(self.gen)

    @classmethod
    def create(cls, *args, device=default_device, **kwargs):
        return cls(DataLoader(*args, **kwargs), device=device)


##############################################################################################################################################

class DataBunch():
    def __init__(self, train_dl, valid_dl, device):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.device = device

    @classmethod
    def create(cls, train_ds, valid_ds, *args, batch_size=64, train_tfm=None, valid_tfm=None, device=None, **kwargs):
        return cls(DeviceDataLoader.create(DatasetTfm(train_ds, train_tfm, *args), batch_size, shuffle=True, device=device, **kwargs),
                   DeviceDataLoader.create(DatasetTfm(valid_ds, valid_tfm, *args), batch_size*2, shuffle=False, device=device, **kwargs),
                   device=device)


##############################################################################################################################################

def sequence2vocab(sequences, vocab_path, ngram=1):
    """
    convert sentences to vocabulary based on the pattern and min count provided
    """
    vocab = {}
    for n, seq in enumerate(sequences):
        if n%10000==0:
            print(n, len(sequences), n/len(sequences))
        word_list = jieba.lcut(seq)
        for i in range(len(word_list)):
            for l in range(1,ngram+1):
                if i + l > len(word_list):
                    continue
                word = ''.join(word_list[i:i+l])
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    print(len(vocab))
    with open(vocab_path, 'w', encoding='utf-8') as file:
        for v in vocab:
            file.write('%s: %s\n'%(v[0],v[1]))


class Tokenizer(object):
    def __init__(self, vocab_file, min_count):
        self.vocab_file = vocab_file
        self.min_count = min_count
        self.vocab = self.load_vocabulary()

    def load_vocabulary(self):
        vocab = {'[padding]': 0}
        inx = 1
        with open(self.vocab_file, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                word, count = line.split(': ')
                if int(count) < self.min_count:
                    break
                vocab[word] = inx
                inx += 1
        vocab['[unknown]'] = len(vocab)
        vocab['[start]'] = len(vocab)
        vocab['[end]'] = len(vocab)
        return vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def convert_tokens_to_ids(self, sequence, max_seq_length, ngram=1):
        tokens = jieba.lcut(sequence)
        ids = [self.vocab['[start]']]
        for i, token in enumerate(tokens):
            for n in range(1, ngram + 1):
                if i + n > len(tokens):
                    break
                w = ''.join(tokens[i:i + n])
                if w not in self.vocab:
                    ids.append(self.vocab['[unknown]'])
                else:
                    ids.append(self.vocab[w])
        ids = ids[:max_seq_length-1]
        ids.append(self.vocab['[end]'])
        while len(ids) < max_seq_length:
            ids.append(self.vocab['[padding]'])
        assert len(ids) == max_seq_length
        return ids
