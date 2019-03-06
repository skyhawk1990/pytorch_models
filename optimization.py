# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from data_util import is_listy, to_device
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

class Learner():
    # 模型训练
    def __init__(self, data, model):
        self.data, self.model = data, to_device(model, data.device)

    def fit(self, epochs, lr, loss_fn=F.cross_entropy, opt_fn=optim.SGD):
        opt = opt_fn(self.model.parameters(), lr=lr)
        fit(epochs, self.model, loss_fn, opt, self.data.train_dl, self.data.valid_dl)

    def accuracy_eval(self):
        accuracy_eval(self.model, self.data.valid_dl)

    def confusion_eval(self, label_map):
        confusion_eval(self.model, self.data.valid_dl, label_map)


def loss_batch(model, xb, yb, loss_fn, opt=None):
    # 计算批次损失函数
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    loss = loss_fn(model(*xb), *yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(yb)


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl):
    # 在train_dl上训练模型，然后在valid_dl上测试
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss, _ = loss_batch(model, xb, yb, loss_fn, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, xb, yb, loss_fn) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


def accuracy_eval(model, valid_dl):
    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for xb, yb in valid_dl:
            correct_num += (torch.argmax(model(xb), dim=1)==yb).float().sum()
            total_num += xb.size(0)
    print('accuracy %s' % (correct_num/total_num))


def confusion_eval(model, valid_dl, label_map):
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in valid_dl:
            preds += torch.argmax(model(xb), dim=1).numpy().tolist()
            labels += yb.numpy().tolist()

    confusion = confusion_matrix(labels, preds)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    # 设置x轴的文字往上走
    ax.set_xticklabels([''] + list(label_map.keys()), rotation=90)
    ax.set_yticklabels([''] + list(label_map.keys()))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
