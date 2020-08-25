import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T



class DQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(DQN, self).__init__()

        self.inut_shape = inputs_shape

        self.conv1 = nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
 
        x = x.reshape(x.size(0),-1) 
        x = F.relu(self.fc4(x))
        return self.head(x)	

 
    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)


