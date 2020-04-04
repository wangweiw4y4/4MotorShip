#!/usr/bin/env python
# coding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
CUDA = torch.cuda.is_available()


class ASVActorNet(nn.Module):
    """
    定义Actor的网络结构：
    三个隐藏层，每层都是全连接，100个神经元
    每层50%的dropout
    隐藏层之间用ReLU激活，输出层使用tanh激活
    """

    def __init__(self, n_states, n_actions, n_neurons=100, a_bound=1):
        """
        @param n_states: number of observations
        @param n_actions: number of actions
        @param n_neurons: 隐藏层神经元数目，按照论文规定是100
        @param a_bound: action的倍率
        """
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(n_states, n_neurons), # 第一个隐藏层
            nn.Dropout(0.5),                # 50%dropout
            nn.LeakyReLU(),                      # ReLu activation
            nn.Linear(n_neurons, n_neurons), # 第二个隐藏层
            nn.Dropout(0.5),                # 50%dropout
            nn.LeakyReLU(),                      # ReLu activation
            nn.Linear(n_neurons, n_neurons), # 第三个隐藏层
            nn.Dropout(0.5),                # 50%dropout
            nn.LeakyReLU(),                      # ReLu activation
            nn.Linear(n_neurons, n_actions),# 输出层
            nn.Tanh(),                      # tanh activation
        )

        self.bound = a_bound
        if CUDA:
            self.bound = torch.FloatTensor([self.bound]).cuda()
        else:
            self.bound = torch.FloatTensor([self.bound])

    def forward(self, x):
        x = x.cuda() if CUDA else x
        action_value = self.layer(x)
        action_value = action_value * self.bound
        return action_value


class ASVCriticNet(nn.Module):
    """定义Critic的网络结构"""

    def __init__(self, n_states, n_actions, n_neurons=64):
        """
        @param n_states: number of observations
        @param n_actions: number of actions
        @param n_neurons: 隐藏层神经元数目
        """
        super(ASVCriticNet, self).__init__()
        self.fc_state = nn.Linear(n_states, n_neurons)
        # self.fc_state.weight.data.normal_(0, 0.1)
        self.fc_action = nn.Linear(n_actions, n_neurons)
        self.fc_q = nn.Linear(2*n_neurons, int(n_neurons/2))
        self.fc_out = nn.Linear(int(n_neurons/2), 1)

    def forward(self, s, a):
        s, a = (s.cuda(), a.cuda()) if CUDA else (s, a)
        h1 = F.relu(self.fc_state(s))
        h2 = F.relu(self.fc_action(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q