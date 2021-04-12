import numpy as np
from collections import namedtuple, deque

class ReplayMemory():
    def __init__(self, capacity=10000, min_capacity=1000):
        self.memory = deque(maxlen=capacity)
        self.min_capacity = min_capacity
        self.Experience = namedtuple('Experience', 
            field_names=['state', 'action', 'reward', 'done', 'next_state']
        )
    
    def append(self, state, action, reward, done, next_state):
        self.memory.append(
            self.Experience(state, action, reward, done, next_state)
        )
    
    def sample(self, batch_size=32):
        samples = np.random.choice(len(self.memory), batch_size, replace=False)
        
        return zip(*[self.memory[i] for i in samples])
    
    def can_provide_sample(self):
        return len(self.memory) > self.min_capacity
