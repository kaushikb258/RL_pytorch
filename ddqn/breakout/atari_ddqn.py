import argparse
import os
import random
import torch
import gym

from torch.optim import Adam
from tester import Tester
from buffer import ReplayBuffer
from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from config import Config
from util import get_class_attr_val
from model import DQN
from trainer import Trainer

#---------------------------------------------------------------------


class DDQNAgent:
    def __init__(self, config: Config, training=True):
        self.config = config
        self.is_training = training
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.model = DQN(self.config.state_shape, self.config.action_dim)
        self.target_model = DQN(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.model.cuda()
        self.target_model.cuda()

    def act(self, state, epsilon=None):
        if (epsilon == None): epsilon = 0.01  
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learn(self, t):
        s, a, r, s2, done = self.buffer.sample(self.config.batch_size)

        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        s2 = torch.tensor(s2, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        s = s.cuda()
        a = a.cuda()
        r = r.cuda()
        s2 = s2.cuda()
        done = done.cuda()

        q_values = self.model(s).cuda()
        next_q_values = self.model(s2).cuda()
        next_q_state_values = self.target_model(s2).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if t % self.config.update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

        
    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)

    def save_checkpoint(self):
        os.makedirs('ckpt', exist_ok=True)
        torch.save(self.model.state_dict(), 'ckpt/model.pt')           

    def load_checkpoint(self):
        self.model.load_state_dict('ckpt/model.pt')
        self.target_model.load_state_dict('ckpt/model.pt')

#------------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    args = parser.parse_args()

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.frames = int(1e7)
    config.learning_rate = 2e-5
    config.max_buff = 500000
    config.update_interval = 10000
    config.batch_size = 32
    config.print_interval = 5000
    config.checkpoint_interval = 50000

    # wrap the env
    env = gym.make(config.env)  #make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape

    

    if args.train:
        agent = DDQNAgent(config)
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        agent = DDQNAgent(config, training=False)
        tester = Tester(agent, env, args.model_path)
        tester.test()

