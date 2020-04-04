#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from asv_nn import ASVActorNet, ASVCriticNet
from Utils import ExpReplay, soft_update, OUProcess
from torch.utils.tensorboard import SummaryWriter
CUDA = torch.cuda.is_available()


class DDPG(object):
    def __init__(self, n_states, n_actions, a_bound=1, lr_a=0.001, lr_c=0.002, tau=0.01, gamma=0.9, 
        noise_var=1.6, n_decay_rate=1, MAX_MEM=10000, MIN_MEM=None, BATCH_SIZE=128, **kwargs):
        # 参数复制
        self.n_states, self.n_actions = n_states, n_actions
        self.tau, self.gamma, self.bound = tau, gamma, a_bound
        self.noise_var = noise_var
        self.noise_decay_rate = n_decay_rate
        self.batch_size = BATCH_SIZE
        # 初始化训练指示符
        self.start_train = False
        self.mem_size = 0
        # 创建经验回放池
        self.memory = ExpReplay(n_states,  n_actions, exp_size=MAX_MEM, exp_thres=MIN_MEM, batch_size=BATCH_SIZE)  # s, a, r, d, s_
        # 创建神经网络并指定优化器
        self._build_net()
        # 创建网络参数记录
        self.all_state = {'actor_eval': self.actor_eval.state_dict(), 'actor_target': self.actor_target.state_dict(), 
        'critic_eval': self.critic_eval.state_dict(), 'critic_target': self.critic_target.state_dict()}
        # 指定summary writer
        if 'summary_path' in kwargs:
            self._build_summary_writer(kwargs['summary_path'])
        else:
            self._build_summary_writer()
        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=lr_a)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=lr_c)
        # 约定损失函数
        self.mse_loss = nn.MSELoss()
        # 开启cuda
        if CUDA:
            self.cuda()

    def _build_net(self):
        n_states, n_actions = self.n_states, self.n_actions
        self.actor_eval = ASVActorNet(n_states, n_actions, a_bound=self.bound)
        self.actor_target = ASVActorNet(n_states, n_actions, a_bound=self.bound)
        self.critic_eval = ASVCriticNet(n_states, n_actions)
        self.critic_target = ASVCriticNet(n_states, n_actions)

    def get_action_noise(self, state):
        """给定当前状态，获取加入衰减噪声的选择动作"""
        action = self.choose_action(state)
        noise_var = self.noise_var * self.noise_decay_rate
        action_noise = np.random.normal(0, noise_var, (1,4))
        action += action_noise
        action = np.clip(action, -self.bound, self.bound)
        return action[0], action_noise

    def choose_action(self, s):
        """给定当前状态，获取选择的动作"""
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.actor_eval.forward(s).detach().cpu().numpy()
        return action

    def learn(self):
        """训练网络"""
        # 将eval网络参数赋给target网络
        soft_update(self.actor_target, self.actor_eval, self.tau)
        soft_update(self.critic_target, self.critic_eval, self.tau)

        # 获取batch并拆解
        batch = self.memory.get_batch_splited_tensor(CUDA, self.batch_size)
        if batch is None:
            return None, None
        else:
            self.start_train = True
        batch_cur_states, batch_actions, batch_rewards, batch_dones, batch_next_states = batch
        # 计算target_q，指导cirtic更新
        # 通过a_target和next_state计算target网络会选择的下一动作 next_action；通过target_q和next_states、刚刚计算的next_actions计算下一状态的q_values
        target_q_next = self.critic_target(batch_next_states, self.actor_target(batch_next_states))
        target_q = batch_rewards + self.gamma * (1 - batch_dones) * target_q_next   # 如果done，则不考虑未来
        # 指导critic更新
        q_value = self.critic_eval(batch_cur_states, batch_actions)
        td_error = self.mse_loss(q_value, target_q)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()

        # 指导actor更新
        policy_loss = self.critic_eval(batch_cur_states, self.actor_eval(batch_cur_states))  # 用更新的eval网络评估这个动作
        # policy_loss = self.critic_eval(batch_cur_states, batch_actions)
        # 如果 a是一个正确的行为的话，那么它的policy_loss应该更贴近0
        loss_a = -torch.mean(policy_loss)
        self.actor_optim.zero_grad()
        loss_a.backward()
        self.actor_optim.step()
        
        return td_error.detach().cpu().numpy(), loss_a.detach().cpu().numpy()
    
    def _build_summary_writer(self, summary_path=None):
        if summary_path:
            self.summary_writer = SummaryWriter(log_dir=summary_path)
        else:
            self.summary_writer = SummaryWriter()
            
    def get_summary_writer(self):
        return self.summary_writer

    def save_model(self, episode):
        self.all_state['noise_decay_rate'] = self.noise_decay_rate
        self.all_state['episode'] = episode
        torch.save(self.all_state, 'models/param %i Episode.pth' % episode)

    def save_beat_model_param(self, episode):
        self.best_model = {'actor_eval': self.actor_eval.state_dict(), 'actor_target': self.actor_target.state_dict(), 
        'critic_eval': self.critic_eval.state_dict(), 'critic_target': self.critic_target.state_dict(), 
        'noise_decay_rate': self.noise_decay_rate, 'episode': episode}

    def load_model(self, episode):
        all_state = torch.load('models/param %i Episode.pth' % episode)
        self.noise_decay_rate = all_state['noise_decay_rate']
        self.actor_eval.load_state_dict(all_state['actor_eval'])
        self.actor_target.load_state_dict(all_state['actor_target'])
        self.critic_eval.load_state_dict(all_state['critic_eval'])
        self.critic_target.load_state_dict(all_state['critic_target'])

    def add_step(self, s, a, r, d, s_):
        self.memory.add_step(s, a, r, d, s_)
        self.mem_size += 1

    def cuda(self):
        self.actor_eval.cuda()
        self.actor_target.cuda()
        self.critic_eval.cuda()
        self.critic_target.cuda()

# if __name__ ==  '_main_':
# print('aaa')
# ddpg = DDPG(6, 4, 4, 0.001, 0.001, 0.01, 0.99)
# ddpg.get_action_noise([1,4.1,0.5,0.01,0.01,0.01])
