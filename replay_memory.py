from collections import deque
import numpy as np

class Memory:
    def __init__(self,size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def sample(self,probabilities=None, batch_size=32):
        l = list(self.buffer)
        # print(l)
        # print(batch_size)
        indices = np.random.choice( len(l), batch_size)
        selected = []
        for i in indices:
            selected.append( l[i] )
        return selected

    def sample_unpack(self, probabilities=None, batch_size=32):
        samples = self.sample(probabilities, batch_size)
        len_cats = len(samples[0])
        categories = []
        for col in range(0,len_cats): categories.append([])
        for col in range(0,len_cats):
            for s in samples:
                categories[col].append(s[col])
        return categories

    def append(self,item):
        self.buffer.append(item)
