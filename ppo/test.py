import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from net import Policy
from ppo import process_frame


env_name = 'BipedalWalker-v2'
lr = 2e-4
    
env = gym.make(env_name)
policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
policy.cuda()
opt = Adam(policy.parameters(), lr=lr)
mse = nn.MSELoss()

policy.load_state_dict(torch.load('ckpt/model.pt'))

policy.eval()


for ep in range(10):

   s = env.reset(); s = process_frame(s)

   done = False
   ep_rewards = 0
   nstep = 0

   while not done:

        env.render()     

        s = Variable(torch.from_numpy(s[np.newaxis])).float().cuda()

        _, action, _, mean_action = policy(s)
        action = mean_action.data.cpu().numpy()[0]

        if (nstep == 0):
           a1 = action
           act = action
        elif (nstep == 1):
           a2 = action
           act = action
        else:
           act = (a1+a2+action)/3.0
           a1 = a2
           a2 = action 

        s2, reward, done, _ = env.step(act)
        ep_rewards += reward
        s2 = process_frame(s2)

        s = s2
        nstep += 1
        
   print('ep: ', ep, ' | ep_rewards: ', ep_rewards, ' | nsteps: ', nstep) 
