import random

class Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory = [None] * memory_size
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.index = 0

    def append(self, transition_tuple):
        """ Parameters:
            transition_tuple: tuple(Current State, Action, Reward, Next State, is Terminal)"""
        self.memory[self.index] = transition_tuple
        self.index = (self.index + 1) % self.memory_size

    def sample_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)
