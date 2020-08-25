import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

#--------------------------------------------------------------------------------

def log_normal_density(x, mean, log_std, std):
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5*np.log(2 * np.pi) - log_std
    log_density = log_density.sum(dim=1, keepdim=True)
    return log_density
#--------------------------------------------------------------------------------

class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super(Policy, self).__init__()

        # actor
        self.act_fc1 = nn.Linear(state_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_space)
        self.mu.weight.data.mul_(0.1)
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # critic
        self.value_fc1 = nn.Linear(state_space, 64)
        self.value_fc2 = nn.Linear(64, 128) 
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)
        

    def forward(self, x):
        # action
        act = F.relu(self.act_fc1(x))
        act = F.relu(self.act_fc2(act))
         
        mean = F.tanh(self.mu(act))  
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # value
        v = F.relu(self.value_fc1(x))
        v = F.relu(self.value_fc2(v))
        v = self.value_fc3(v)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

#--------------------------------------------------------------------------------
