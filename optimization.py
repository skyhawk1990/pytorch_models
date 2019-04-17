# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from data_util import is_listy, to_device
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False


class Learner():
    # 模型训练
    def __init__(self, data, model):
        self.data, self.model = data, to_device(model, data.device)

    def fit(self, epochs, init_lr, decay_rate=1, decay_epoch=1, loss_fn=F.cross_entropy, opt_fn=optim.SGD):
        # SGD 随机梯度下降
        # opt = torch.optim.SGD(model.parameters(), lr=init_lr)
        # momentum 动量加速, 在SGD函数里指定momentum的值即可
        # opt = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
        # RMSprop 指定参数alpha
        # opt = torch.optim.RMSprop(model.parameters(), lr=init_lr, alpha=0.9)
        # Adam 参数betas=(0.9, 0.99)
        # opt = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.99))
        opt = opt_fn(self.model.parameters(), lr=init_lr)
        fit(init_lr, decay_rate, decay_epoch, epochs, self.model, loss_fn, opt, self.data.train_dl, self.data.valid_dl)

    def accuracy_eval(self):
        accuracy_eval(self.model, self.data.valid_dl)

    def confusion_eval(self, label_map):
        confusion_eval(self.model, self.data.valid_dl, label_map)


def loss_batch(model, xb, yb, loss_fn, opt=None):
    # 计算批次损失函数
    logits = model(xb)
    preds = torch.argmax(logits, dim=1)
    correct_num = (preds==yb).float().sum()
    total_num = len(yb)

    loss = loss_fn(logits, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), correct_num, total_num


def fit(init_lr, decay_rate, decay_epoch, epochs, model, loss_fn, opt, train_dl, valid_dl):
    # 在train_dl上训练模型，然后在valid_dl上测试
    step = 0
    for epoch in range(epochs):
        lr = init_lr * (decay_rate ** (epoch//decay_epoch))
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        model.train()
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, xb, yb, loss_fn, opt)
            step += 1
            if step%500 == 0:
                print('train loss:', step, loss)

        model.eval()
        with torch.no_grad():
            losses, correct_nums, total_nums = zip(*[loss_batch(model, xb, yb, loss_fn) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, total_nums)) / np.sum(total_nums)
        accuracy = np.sum(correct_nums)/np.sum(total_nums)

        print('valid:', epoch, lr, val_loss, accuracy)


def accuracy_eval(model, valid_dl):
    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for xb, yb in valid_dl:
            correct_num += float((torch.argmax(model(xb), dim=1)==yb).float().sum())
            total_num += float(xb.size(0))
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
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

    plt.show()
