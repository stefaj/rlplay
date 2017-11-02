from collections import deque
import numpy as np

class Memory:
    def __init__(self,size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def sample(self,probabilities, batch_size=32):
        l = list(self.buffer)
        # print(l)
        # print(batch_size)
        indices = np.random.choice( len(l), batch_size)
        selected = []
        for i in indices:
            selected.append( l[i] )
        return selected

    def append(self,item):
        self.buffer.append(item)
