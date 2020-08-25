import os
import gym
import numpy as np
import torch
import torch.nn as nn
import sys
import time

from itertools import count

from ppo import ppo_update, generate_trajectory
from torch.optim import Adam
from net import Policy

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#-------------------------------------------------------------------------
parallel = True
if (size == 1):
   parallel = False
#-------------------------------------------------------------------------


lr = 1e-4
batch_size = 64
nupdates = 5
nepoch = 40000
clip_value = 0.1
gamma = 0.99    
max_tsteps = 2048 

load_weights = True #False
ep_start = 2960  #0


if (not load_weights):
    ep_start = 0

#------------------------------------------------------------------------

env = gym.make('BipedalWalker-v2') 

policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])

if (load_weights):
   policy.load_state_dict(torch.load('ckpt/model.pt'))
policy.cuda()

opt = Adam(policy.parameters(), lr=lr)



for ep in range(ep_start, nepoch):
           
        observations, actions, rewards, values, logprobs, gaes, returns = generate_trajectory(gamma, env, policy, max_tsteps, is_render=False) 
        
        if (rank != 0 and parallel):
             data = [observations, actions, rewards, values, logprobs, gaes, returns]
             comm.send(data, dest=0, tag=rank) 
        else:
             if (parallel):
                 for proc in range(1, size):
                     data = comm.recv(source=proc, tag=proc)
                     obs, act, rew, val, logp, gae, ret = data

                     observations = np.concatenate([observations,obs],0) 
                     actions = np.concatenate([actions,act],0)
                     rewards = np.concatenate([rewards,rew],0)  
                     values = np.concatenate([values,val],0)
                     logprobs = np.concatenate([logprobs,logp],0)
                     gaes = np.concatenate([gaes,gae],0)
                     returns = np.concatenate([returns,ret],0)
                  

             memory = (observations, actions, values, logprobs, gaes, returns)

             ppo_update(policy, opt, batch_size, memory, nupdates, gamma, clip_value=clip_value)

             print('epoch: ', ep, ' | epoch rewards: ', rewards.sum(), ' | num steps: ', rewards.shape[0])
       
             f = open('performance.txt', 'a+')
             f.write(str(ep) + ' ' + str(rewards.sum()/float(size)) + '\n')
             f.close()


             if (ep % 10 == 0):
                torch.save(policy.state_dict(), 'ckpt/model.pt')

 
        if (parallel):
             comm.Barrier()
        
             # broadcast neural net weights from 0 to everyone
             with torch.no_grad():

                 policy.cpu()

                 for param_tensor in policy.state_dict():
                  
                      weights = policy.state_dict()[param_tensor]

                      if (rank == 0):
                          for proc in range(1, size):
                             comm.send(weights, dest=proc, tag=proc)  
                      else:
                          weights = comm.recv(source=0, tag=rank)  

                      if (rank != 0):
                          policy.state_dict()[param_tensor].data.copy_(weights) 

                 policy.cuda()
            
#----------------------------------------------------------------------------
