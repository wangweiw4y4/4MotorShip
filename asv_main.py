#!/usr/bin/env python
# coding=utf-8

from asv_agent import DDPG
from asv_env import ASVEnv
import numpy as np
import torch
import json

def rl_loop(load_model_episode=0, save_model=True):
    #ENV_NAME = 'ASV_LINE'
    # RENDER = False
    MAX_EPISODES = 40000
    MAX_EP_STEPS = 300

    LR_A = 0.0005
    LR_C = 0.001
    LR_SOFT_UPDATE = 0.005
    GAMMA = 0.99
    NOISE_VAR = 1.6
    NOISE_DECAY = 0.995
    MAX_MEM=10000
    MIN_MEM=None
    BATCH_SIZE=128

    MODEL_INTERVAL = 500
    TRACE_INTERVAL = 50
    REWARD_INTERVAL = 20

    env = ASVEnv()
    # env = gym.make(ENV_NAME)
    # env = env.unwrapped
    # env.seed(1)
    s_dim = env.observation_space['shape']
    a_dim = env.action_space['shape']
    a_bound = env.action_space['high']

    ddpg = DDPG(s_dim, a_dim, a_bound, LR_A, LR_C, LR_SOFT_UPDATE, GAMMA, NOISE_VAR, 1, MAX_MEM, MIN_MEM, BATCH_SIZE)
    
    if load_model_episode != 0:
        ddpg.load_model(load_model_episode)

    ep_reward_log = []
    ep_reward_tb = 0
    best_ep_r = -500000
    best_ep = 0

    for i in range(load_model_episode, MAX_EPISODES+load_model_episode):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            # if RENDER:
            #     env.render()
            a, action_noise = ddpg.get_action_noise(s)
            s_, r ,done = env.step(a)
            ddpg.add_step(s, a, r, done, s_)
            s = s_
            error = s[1]-s[0]   # 得到的新状态s点和目标轨迹点的差

            ep_reward += r
            ep_reward_tb += r
            # 每隔TRACE_INTERVAL局记录 整局全数据
            if i % TRACE_INTERVAL == 0:
                print('Episode: ', i, ' Reward: ', '%.2f' % r, ' Error: ', '%.2f' % error, ' X: ', '%.2f' % s[0], ' Y: ', '%.2f' % s[1], 
                    ' f1: ', '%.2f' % a[0], ' f2: ', '%.2f' % a[1], ' f3: ', '%.2f' % a[2], ' f4: ', '%.2f' % a[3], ' A_noise: ', '%.4f' % action_noise)
            # 记录每局的ep_reward
            if j == MAX_EP_STEPS-1:
                ep_reward_log.append(int(ep_reward))
                # print('Episode:', i, ' Reward: %i' % int(ep_reward))
                # if ep_reward > -300:
                #     RENDER = Truegit
                break
        
        if save_model == True:
            # 记录ep_reward最大的最优对局局次,保存此时的模型参数
            if ep_reward >= best_ep_r and ddpg.mem_size > MAX_MEM // 10:
                best_ep_r = ep_reward
                best_ep = i
                ddpg.save_beat_model_param(best_ep)
                if i == (MAX_EPISODES+load_model_episode-1):
                    torch.save(ddpg.best_model, 'models/best %i Episode.pth' % best_ep)
            # 每MODEL_INTERVAL局 保存模型参数
            if (i+1) % MODEL_INTERVAL == 0 :
                ddpg.save_model(i+1)
        
        # 训练：每局训练10次
        if ddpg.mem_size > MAX_MEM // 10:
            for k in range(10):
                td_error, loss_a = ddpg.learn()
                ddpg.noise_decay_rate *= NOISE_DECAY    #开始训练之后，每次训练后噪声衰减一次
                # 记录每步学习的loss,每局10个记录点
                ddpg.summary_writer.add_scalar('td_error', td_error, k+10*i)
                ddpg.summary_writer.add_scalar('loss_a', loss_a, k+10*i)
                # ddpg.summary_writer.add_histogram('Actor_eval/')

        # 每REWARD_INTERVAL局 计算平均单步reward，tensorboard可视化
        if (i+1) % REWARD_INTERVAL ==0:
            ddpg.summary_writer.add_scalar('reward', ep_reward_tb/(REWARD_INTERVAL), i)
            ep_reward_tb = 0

    json_str = json.dumps(ep_reward_log)
    with open('ep_reward_log.json', 'w') as json_file:
        json_file.write(json_str)

if __name__ == '__main__':
    rl_loop(save_model=False)

    