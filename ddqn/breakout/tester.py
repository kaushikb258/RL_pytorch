import time


class Tester(object):

    def __init__(self, agent, env, model_path, num_episodes=50):

        self.num_episodes = num_episodes
        self.agent = agent
        self.env = env
        self.agent.load_weights(model_path)
        self.policy = lambda x: agent.act(x)

    def test(self, visualize=True):
        avg_reward = 0

        for ep in range(self.num_episodes):

            s = self.env.reset()
            tsteps = 0
            ep_reward = 0.
            done = False
            ale_lives = 5

            while not done:

                if visualize:
                    self.env.render()
                    

                action = self.policy(s)
                s, reward, done, info = self.env.step(action)

                ep_reward += reward
                tsteps += 1
 
                if (info['ale.lives'] == ale_lives-1):
                      # lost a life
                      ale_lives = info['ale.lives']
                      s, _, done, info = self.env.step(1) 

                time.sleep(0.02) 

            print('test episode: ', ep, ' | ep reward: ', ep_reward)

            avg_reward += ep_reward

        avg_reward /= self.num_episodes
        
        print("avg reward: ", avg_reward)
        print('done')
        print('-'*20)




