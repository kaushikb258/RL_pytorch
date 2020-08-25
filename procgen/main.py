import argparse
import ctypes
import multiprocessing as mp

import numpy as np
import torch

from model import CNNModel, ImpalaModel, Discrete, Normal 
from train import train_step, gae
from worker import GamePlayer

#-------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', default=.99, type=float)
parser.add_argument('--lam', default=.95, type=float)
parser.add_argument('--epsilon', default=.1, type=float)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--num_iterations', default=int(2e8), type=int)
parser.add_argument('--num_steps', default=256, type=int)
parser.add_argument('--ppo_epochs', default=4, type=int)
parser.add_argument('--num_batches', default=4, type=int)
parser.add_argument('--lr', default=2.5e-4, type=float)
args = parser.parse_args()

args.batch_size = int(args.num_workers / args.num_batches)
args.num_actions = 15
args.obs_shape = (64, 64)
action_type = 'discrete'
scalar_shape = (args.num_workers, args.num_steps)

batch_obs_shape = (args.num_workers, args.num_steps, 1, 64, 64)

shared_obs_c = mp.Array(ctypes.c_float, int(np.prod(batch_obs_shape)))
shared_obs = np.frombuffer(shared_obs_c.get_obj(), dtype=np.float32)
shared_obs = np.reshape(shared_obs, batch_obs_shape)

rewards = np.zeros(scalar_shape, dtype=np.float32)
discounted_rewards = np.zeros(scalar_shape, dtype=np.float32)
episode_ends = np.zeros(scalar_shape, dtype=np.float32)
values = np.zeros(scalar_shape, dtype=np.float32)
policy_probs = np.zeros(scalar_shape, dtype=np.float32)

actions = np.zeros(scalar_shape, dtype=np.int32)

game_player = GamePlayer(args, shared_obs)

dist = Discrete(args.num_actions) # discrete actions
#dist = Normal(args.num_actions) # continuous actions

# 1 channel input
#model = CNNModel(1, args.num_actions, dist).to('cuda')
model = ImpalaModel(1, args.num_actions, dist).to('cuda')


optim = torch.optim.Adam(model.parameters(), lr=args.lr)


for iter_step in range(args.num_iterations):
    game_player.run_rollout(args, shared_obs, rewards, discounted_rewards,
           values, policy_probs, actions, model, episode_ends)

    observations = shared_obs.copy()

    advantages = gae(rewards, values, episode_ends, args.gamma, args.lam)
    advantages = advantages.astype(np.float32)
    rewards_to_go = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    if (iter_step % 10 == 0):
        ep_rew = np.mean(np.array(list(game_player.episode_rewards)))
        print('iter step: ', iter_step, ' | mean ep rewards: ', ep_rew)
        f = open('performance.txt', 'a+')
        f.write(str(iter_step) + ' ' + str(ep_rew) + '\n')
        f.close()

    if (iter_step % 1000 == 0):
        print('saving model ')
        torch.save(model.state_dict(), 'ckpt/model.pt')


    for batch in range(args.num_batches):
        start = batch * args.batch_size
        end = (batch + 1) * args.batch_size

        train_data = (advantages, rewards_to_go, values, actions, observations, policy_probs)
        batch_data = [x[start:end] for x in train_data]
        batch_data = [torch.tensor(x).to('cuda') for x in batch_data]

        batch_data = [x.reshape((-1, ) + x.shape[2:]) for x in batch_data]

        for epoch in range(args.ppo_epochs):
            train_step(model, optim, batch_data, args)

#------------------------------------------------------------------------------------------

