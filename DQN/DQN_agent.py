import gym
import numpy as np
import pdb

from  DQN.replay_memory import Replay_Memory

class DQN_Agent():

    def __init__(self, environment_name, memory_size = 50000, burn_in = 10000, render=False):

            # Create an instance of the network itself, as well as the memory.
            # Here is also a good place to set environmental parameters,
            # as well as training parameters - number of episodes / iterations, etc.
            self.env = gym.make(environment_name)
            self.init_state = self.env.reset()
            self.replay_memory = Replay_Memory(memory_size = memory_size, burn_in = burn_in)

    def epsilon_greedy_policy(self, q_values):
            # Creating epsilon greedy probabilities to sample from.
            pass

    def greedy_policy(self, q_values):
            # Creating greedy policy for test time.
            pass

    def train(self):
            # In this function, we will train our network.
            # If training without experience replay_memory, then you will interact with the environment
            # in this function, while also updating your network parameters.

            # If you are using a replay memory, you should interact with environment here, and store these
            # transitions to memory, while also updating your model.
            pass

    def test(self, model_file=None):
            # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
            # Here you need to interact with the environment, irrespective of whether you are using a memory.
            pass

    def burn_in_memory(self):
            # Initialize your replay memory with a burn_in number of episodes / transitions.
            count = 0
            while count != self.replay_memory.burn_in:
                state = self.env.reset()
                for t in range(200):
                    action = self.env.action_space.sample()
                    next_state, reward, done, info = self.env.step(action)
                    transition_tuple = (state, action, reward, next_state, done)
                    self.replay_memory.append(transition_tuple)
                    count += 1
                    state = next_state
                    if count == self.replay_memory.burn_in or done:
                        break


