import random

class Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = []
        self.index = 0

    def append(self,transition_tuple):
        """ Parameters:
            transition_tuple: tuple(S,A,R,S',done)"""

        if self.index < self.memory_size:
            self.memory.append(transition_tuple)
            index += 1
        else:
            self.memory[index % self.memory_size] = transition_tuple
            index += 1

    def sample_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)

