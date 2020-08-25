import numpy as np
import torch

def train_step(model, optim, batch_data, args):
    model.train()
    optim.zero_grad()

    advantages, rewards_to_go, values, actions, obs, selected_prob = batch_data

    values_new, dist_new = model(obs)
    values_new = values_new.flatten()
    selected_prob_new = dist_new.log_prob(actions)

    # pi ratio loss
    prob_ratio = torch.exp(selected_prob_new) / torch.exp(selected_prob)
    surr1 = prob_ratio * advantages
    surr2 = torch.clamp(prob_ratio, 1 - args.epsilon, 1 + args.epsilon) * advantages
    ppo_loss = -torch.mean(torch.min(surr1, surr2))

    # value loss
    value_loss = torch.mean((values_new - rewards_to_go).pow(2))

    # entropy loss
    entropy_loss = torch.mean(dist_new.entropy())

    loss = ppo_loss + (0.5)*value_loss - (0.01)*entropy_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    




def gae(rewards, values, episode_ends, gamma, lam):

    episode_ends = 1.0 - episode_ends
    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        advantages[:, t] = gae_step
    return advantages
