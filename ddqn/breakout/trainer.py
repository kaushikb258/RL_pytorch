import math
import numpy as np
from config import Config


def get_epsilon(t):
    if (t <= int(1e6)):
                epsilon = 1.0 - (1.0 - 0.1)/(1.0e6)*float(t)
    elif (t > int(1e6) and t <= int(2e6)):
                epsilon = 0.1 - (0.1 - 0.01)/(1.0e6)*(float(t)-1e6)
    else:
                epsilon = 0.01
    epsilon = min(max(epsilon, 0.01), 1.0)  
    return epsilon 



class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config
        
    def train(self, pre_fr=0):
        all_rewards = []       

        state = self.env.reset()
                 

        t = pre_fr 

        for ep in range(50000):

          nsteps = 0
          ale_lives = 5   
          ep_reward = 0
          time_to_fire = True   
          done = False 

          while not done:


            t += 1 

            epsilon = get_epsilon(t)               

            # random noops followed by one fire
            if (time_to_fire):
                 # random noop
                 for _ in range(np.random.randint(low=0,high=7)):
                    state, _, _, _ = self.env.step(0)  
                 # FIRE 
                 action = 1
                 time_to_fire = False 
            else:  
                 action = self.agent.act(state, epsilon)
             
            next_state, reward, done, info = self.env.step(action)

            ale_lives2 = int(info['ale.lives'])
            
            assert ale_lives2 <= 5
            assert ale_lives2 == ale_lives or ale_lives2 == ale_lives-1

            if (ale_lives2 != ale_lives):
                  # life lost
                  self.agent.buffer.add(state, action, reward, next_state, True)
                  ale_lives = ale_lives2
                  time_to_fire = True
            else:
                  # life not lost
                  self.agent.buffer.add(state, action, reward, next_state, done)
 

            state = next_state
            ep_reward += reward 
            nsteps += 1

            loss = 0

            if (self.agent.buffer.size() > self.config.batch_size):
                loss = self.agent.learn(t)
 
                
            if (t % self.config.print_interval == 0 and len(all_rewards) > 25):
                print('step: ', t, ' | mean reward (last 25 episodes): ', np.mean(all_rewards[-25:]), ' | loss: ', loss, ' | episode: ', ep) 
                

            if (t % self.config.checkpoint_interval == 0):
                self.agent.save_checkpoint()
              
  
            if done:
                state = self.env.reset()
                all_rewards.append(ep_reward)
                
                
