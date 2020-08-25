from collections import deque
import multiprocessing as mp
import sys
import gym
import numpy as np
import torch
from torchvision import transforms



class GamePlayer:
    """A manager class for running multiple game-playing processes."""
    def __init__(self, args, shared_obs):
        self.episode_length = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)

        self.processes = []
        for i in range(args.num_workers):
            parent_conn, child_conn = mp.Pipe()
            worker = SubprocWorker(i, child_conn, args, shared_obs)
            ps = mp.Process(target=worker.run)
            ps.start()
            self.processes.append((ps, parent_conn))

    def run_rollout(self, args, shared_obs, rewards, discounted_rewards, values,
                    policy_probs, actions, model, episode_ends):
        model.eval()
        step_actions = actions[:, -1]
        shared_obs[:, 0] = shared_obs[:, -1]
        for step in range(args.num_steps):
            obs = shared_obs[:, step]
            
            obs_torch = torch.tensor(obs).to('cuda').float()
            step_values, dist = model(obs_torch)

            step_actions = dist.sample()
            step_policy_probs = dist.log_prob(step_actions)

            step_actions = step_actions.detach().cpu().numpy()
            values[:, step] = step_values.detach().cpu().numpy().flatten()
            policy_probs[:, step] = step_policy_probs.detach().cpu().numpy()
            actions[:, step] = step_actions

            for j, (p, pipe) in enumerate(self.processes):
                pipe.send(("step", step, step_actions[j]))

            for j, (p, pipe) in enumerate(self.processes):
                (reward, discounted_reward, done, info) = pipe.recv()
                rewards[j, step] = reward
                discounted_rewards[j, step] = discounted_reward
                episode_ends[j, step] = done
                try:
                    self.episode_length.append(info['final_episode_length'])
                    self.episode_rewards.append(info['final_episode_rewards'])
                except KeyError:
                    pass


class SubprocWorker:
    """A worker for running an environment, intended to be run on a separate
    process."""
    def __init__(self, index, pipe, args, shared_obs):
        self.index = index
        self.pipe = pipe
        self.episode_steps = 0
        self.episode_rewards = 0
        self.disc_ep_rewards = 0
        self.previous_lives = 0
        self.args = args
        self.shared_obs = shared_obs

        self.env = gym.make("procgen:procgen-starpilot-v0", start_level=0, num_levels=0, use_sequential_levels=True, distribution_mode="easy")  
        self.env.reset()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),transforms.Grayscale(),
            transforms.Resize((64, 64)), transforms.ToTensor(),])

    def run(self):
        try:
            while True:
                cmd, t, action = self.pipe.recv()
                if cmd == 'step':
                    self.pipe.send(self.step(action, t))
                elif cmd == 'close':
                    self.pipe.send(None)
                    break
                else:
                    raise RuntimeError('Got unrecognized cmd %s' % cmd)
        except KeyboardInterrupt:
            print('worker: got KeyboardInterrupt')
        finally:
            self.env.close()

    def step(self, action, t):
        step_reward = 0
        for _ in range(1): #self.args.steps_to_skip):
            obs, reward, done, info = self.env.step(action)
            fake_done = done

            self.episode_rewards += reward
            step_reward += reward

            if done:
                info["final_episode_length"] = self.episode_steps
                info["final_episode_rewards"] = self.episode_rewards
                obs = self.env.reset()
                self.episode_steps = 0
                self.episode_rewards = 0

            self.episode_steps += 1

            if t < self.args.num_steps - 1:
                obs = self.transform(obs).numpy()
                self.shared_obs[self.index, t+1] = obs

            if done or fake_done:
                break

    
        self.disc_ep_rewards = self.disc_ep_rewards * self.args.gamma + step_reward
        last_disc_reward = self.disc_ep_rewards
        if done or fake_done:
            self.disc_ep_rewards = 0

        return step_reward, last_disc_reward, fake_done, info
