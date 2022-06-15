#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 03_RNN.py
# @Author: Stormzudi
# @Date  : 2021/5/26 23:42


"""
实习简单的RNN模型，序列生成

# 背景
有 n 句话，每句话都由且仅由 3 个单词组成。要做的是，将每句话的前两个单词作为输入，
最后一词作为输出，训练一个 RNN 模型

"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor





# 定义输入数据
sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
n_class = len(vocab)




# TextRNN Parameter
# 预处理数据
# 构建 Dataset，定义 DataLoader，输入数据用 one-hot 编码
batch_size = 2
# n_step = 2 # number of cells(= number of Step) # 输入是2个单词，所以两步
n_hidden = 5 # number of hidden units in one cell

def make_data(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word2idx[n] for n in word[:-1]]
        target = word2idx[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


input_batch, target_batch = make_data(sentences)
input_batch, target_batch = torch.Tensor(input_batch), torch.LongTensor(target_batch)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)

        # 设置网络的全连接层，输入的二维张量的大小，即输入的[batch_size, size]中的size
        # 输出的二维张量的形状为[batch_size，output_size]中的output_size
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, hidden, X):
        # X: [batch_size, n_step, n_class]
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        out, hidden = self.rnn(X, hidden)
        # out : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        out = out[-1] # [batch_size, num_directions(=1) * n_hidden] ⭐
        model = self.fc(out)
        return model




