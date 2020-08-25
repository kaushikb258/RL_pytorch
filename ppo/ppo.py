import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import copy
import sys
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

#--------------------------------------------------------------------------------------

def calculate_gaes(rewards, dones, values, last_value, gamma, lam_gae=1.0):

    
    # idiot check
    ndones = 0
    for i in range(dones.shape[0]):
       if (dones[i]):
           ndones += 1
    assert ndones == 1, print('ndones: ', ndones)


    assert rewards.shape[0] == values.shape[0], print(rewards.shape, values.shape)

    values_tp1 = np.zeros((values.shape),dtype=np.float32)
    values_tp1[:-1] = values[1:]; values_tp1[-1] = last_value 
    
    assert rewards.shape[0] == values_tp1.shape[0], print(rewards.shape, values.shape, values_tp1.shape)

    deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(rewards, values_tp1, values)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
       gaes[t] = gaes[t] + (gamma*lam_gae) * gaes[t + 1]    

    gaes = np.array(gaes)


    # returns = r + gamma*v*(1-dones)

    returns = np.zeros(rewards.shape[0])

    
    assert rewards.shape[0] == values.shape[0]
    assert rewards.shape[0] == dones.shape[0]

    for t in range(returns.shape[0]):
        returns[t] = rewards[t] + gamma*(1.0 - float(dones[i]))*values_tp1[t]

    return gaes, returns 

#--------------------------------------------------------------------------------------

def ppo_update(policy, optimizer, batch_size, memory, nupdates, gamma, clip_value=0.2):
 
    s, actions, values, logprobs, gaes, returns = memory

    nsamples = gaes.shape[0]  

    s = s[:nsamples]
    actions = actions[:nsamples]
    values = values[:nsamples] 
    logprobs = logprobs[:nsamples]
    returns = returns[:nsamples]     

    gaes = (gaes - gaes.mean()) / gaes.std()

    for update in range(nupdates):
        sampler = BatchSampler(SubsetRandomSampler(list(range(gaes.shape[0]))), batch_size=batch_size, drop_last=False)

        for _, index in enumerate(sampler):

            sampled_states = Variable(torch.from_numpy(s[index])).float().cuda()
            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_returns = Variable(torch.from_numpy(returns[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(gaes[index])).float().cuda()
            

            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_states, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_returns = sampled_returns.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_returns)

            loss = policy_loss + 0.5*value_loss - (1e-3)*dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return value_loss.data.item(), policy_loss.data.item(), dist_entropy.data.item()

#--------------------------------------------------------------------------------------

def process_frame(s):
   s = np.array(s,dtype=np.float32)
#   s = s/255.0
#   s = s[:175,50:225,:3]
#   s = np.transpose(s,(2,0,1))
   return s

#--------------------------------------------------------------------------------------

def generate_trajectory(gamma, env, policy, max_tsteps, is_render=False):
 
    nstep = 0
    s = env.reset(); s = process_frame(s) 
    done = False
    states, actions, rewards, logprobs, dones, values = [], [], [], [], [], []


    while not done:

        if is_render:
            env.render()

        s = Variable(torch.from_numpy(s[np.newaxis])).cuda()

        value, action, logprob, _ = policy(s)

        value = value.data.cpu().numpy()[0]
        action = action.data.cpu().numpy()[0]
        logprob = logprob.data.cpu().numpy()[0]

        s2, reward, done, _ = env.step(action)
        s2 = process_frame(s2) 

        # taking too long, force termination of episode
        if (nstep >= max_tsteps):
            done = True


        states.append(s.data.cpu().numpy()[0])
        actions.append(action)
        rewards.append(reward)
        logprobs.append(logprob)
        dones.append(done)
        values.append(value[0])
        

        s = s2
        nstep += 1
   

    if done:
        last_value = 0.0 
    else:
        print('====== something wrong ', done)
        sys.exit()  
#        s = Variable(torch.from_numpy(s[np.newaxis])).cuda()
#        value, action, logprob, _ = policy(s)
#        print(value.shape)
#        last_value = value.data[0][0]

    states = np.asarray(states) 
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    logprobs = np.asarray(logprobs)
    dones = np.asarray(dones)
    values = np.asarray(values)
    
    gaes, returns = calculate_gaes(rewards, dones, values, last_value, gamma) 

    return states, actions, rewards, values, logprobs, gaes, returns

#--------------------------------------------------------------------------------------

