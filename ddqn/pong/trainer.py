import math
import numpy as np
from config import Config


class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda t: epsilon_final + (epsilon_start-epsilon_final) * math.exp(-1.*t/epsilon_decay)


    def train(self, pre_fr=0):
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0

        state = self.env.reset()

        for t in range(pre_fr + 1, self.config.frames + 1):

            epsilon = self.epsilon_by_frame(t)
            action = self.agent.act(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            loss = 0

            if (self.agent.buffer.size() > self.config.batch_size):
                loss = self.agent.learn(t)
                losses.append(loss)
                
            if (t % self.config.print_interval == 0):
                print('step: ', t, ' | mean reward (last 25 episodes): ', np.mean(all_rewards[-25:]), ' | loss: ', loss, ' | ep num: ', ep_num) 
                

            if (t % self.config.checkpoint_interval == 0):
                self.agent.save_checkpoint()
              
  
            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                
