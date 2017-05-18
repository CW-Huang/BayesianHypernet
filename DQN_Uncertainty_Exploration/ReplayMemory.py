from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, max_size=128):
        self.memory = deque(maxlen=max_size)

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        idxs = np.random.choice(len(self.memory), batch_size)
        return [self.memory[idx] for idx in idxs]

    def add(self, item):
        self.memory.append(item)