import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------------------------------------------------

def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

#----------------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaModel(nn.Module):
    def __init__(self, in_channels, num_outputs, dist, hidden_size=512):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(32 * 8 * 8, hidden_size)
  
        self.critic = nn.Linear(hidden_size, 1)
        self.actor = nn.Linear(hidden_size, num_outputs)
        self.dist = dist
        
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(x)
         
        # flattem
        x = x.view(x.size(0), -1)
        
        # fc
        x = F.relu(self.fc(x))
        
        # value 
        value = self.critic(x)

        # actor
        action_logits = self.actor(x)

        return value, self.dist(action_logits)

#----------------------------------------------------------------------------------

class CNNModel(nn.Module):
    def __init__(self, num_channels, num_outputs, dist, hidden_size=512):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(128 * 10 * 10, hidden_size)

        self.critic = nn.Linear(hidden_size, 1)
        self.actor = nn.Linear(hidden_size, num_outputs)
        self.dist = dist

        self.apply(xavier_uniform_init)


    def forward(self, x):
        # x: [8, 1, 64, 64]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))


        # flatten
        x = x.view(x.size(0), -1)

        # fc1
        x = F.relu(self.fc1(x))         

        # value 
        value = self.critic(x)

        # actor
        action_logits = self.actor(x)

        return value, self.dist(action_logits)

#----------------------------------------------------------------------------------

# used for discrete actions
class Discrete(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 3:
            probs = nn.functional.softmax(x, dim=2)
        elif len(x.shape) == 2:
            probs = nn.functional.softmax(x, dim=1)
        else:
            print(x.shape)
            raise
        dist = torch.distributions.Categorical(probs=probs)
        return dist

# used for continuous actions
class Normal(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.stds = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        dist = torch.distributions.Normal(loc=x, scale=self.stds.exp())
        dist.old_log_prob = dist.log_prob
        dist.log_prob = lambda x: dist.old_log_prob(x).sum(-1)
        return dist


#----------------------------------------------------------------------------------

