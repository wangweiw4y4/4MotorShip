#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math

class ASVEnv:
    """
    AUV 的环境
    AUV的状态(state)由六个部分组成，分别是：
        当前坐标x
        当前坐标y
        当前夹角theta
        转机1速度 u
        转机2速度 v
        转机3速度 r
    AUV的动作(action)是控制四个转机速度的列表[]
    AUV的状态变化通过调用c语言接口实现
    """
    def __init__(self):
        self.x, self.y = 0, 0
        self.theta = 0
        self.u, self.v, self.r = 0, 0, 0
        self.observation_space = {'shape' : 6, 'low' : -50, 'high' : 50}
        self.action_space = {'shape' : 4, 'low' : -10, 'high' : 10}
        # self.observation_space = spaces.Box(shape=(6, ), low=-50, high=50)
        # self.action_space = spaces.Box(shape=(4, ), low=-4, high=4)
    
    def reset(self):
        self.x, self.y = 0, 0
        self.theta = 0
        self.u, self.v, self.r = 0, 0, 0

        return self.get_obs()

    def get_obs(self):
        obs = np.array([self.x, self.y, self.theta, self.u, self.v, self.r])
        return obs
    def get_done(self):
        # 判断本局游戏是否结束/死亡，这里由于没有死亡一说，暂时不做处理
        done = False
        return done
        
    def get_reward(self, s, a):
        x_target = float('%.1f' % (s[0] + 0.1))  # 目标x点取 s的x + 0.1
        y_target = x_target
        reward = -math.log(100*abs(self.x-x_target)) - math.log(100*abs(self.y-y_target)) - 1e-6*(a[0]**2 + a[1]**2 + a[2]**2 + a[3]**2)
        # reward = -(self.x-x_target)**2 - (self.y-y_target)**2 - 1e-6*(a[0]**2 + a[1]**2 + a[2]**2 + a[3]**2)
        return reward
        
    def step(self, action):
        from c_env.step import step
        cur_state = self.get_obs()
        next_state = step(cur_state, action) 
        
        # 将状态更新到变量中
        self.x, self.y, self.theta, self.u, self.v, self.r = next_state
        obs = self.get_obs()
        reward = self.get_reward(cur_state, action)
        done = self.get_done()
        return obs, reward, done