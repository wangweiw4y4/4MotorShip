import torch
from asv_env import ASVEnv
import numpy as np
from asv_nn import ASVActorNet, ASVCriticNet
CUDA = torch.cuda.is_available()

def test(load_episode):
    MAX_EP_STEPS = 300

    env = ASVEnv()
    s_dim = env.observation_space['shape']
    a_dim = env.action_space['shape']
    a_bound = env.action_space['high']

    actor_eval = ASVActorNet(s_dim, a_dim, a_bound=a_bound)
    critic_eval = ASVCriticNet(s_dim, a_dim)
    if CUDA:
        actor_eval.cuda()
        critic_eval.cuda()

    all_state = torch.load('models/param %i Episode.pth' % load_episode)
    actor_eval.load_state_dict(all_state['actor_eval'])
    critic_eval.load_state_dict(all_state['critic_eval'])

    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        a = actor_eval.forward(s).detach().cpu().numpy()[0]
        s_, r ,done, s_target = env.step(a, j)
        ep_reward += r
        s = s_
        error = (s[0]-s_target[0])**2 + (s[1]-s_target[1])**2

        print('Step: ', j, ' Reward: ', '%.2f' % r, ' Error: ', '%.2f' % error, ' X: ', '%.2f' % s[0], ' Y: ', '%.2f' % s[1], ' X_t: ', s_target[0], ' Y_t: ', s_target[1], ' f1: ', '%.2f' % a[0], ' f2: ', '%.2f' % a[1], ' f3: ', '%.2f' % a[2], ' f4: ', '%.2f' % a[3])
    print('Training episode:', load_episode, '    Episode reward:', ep_reward)
if __name__ == '__main__':
    test(15350)