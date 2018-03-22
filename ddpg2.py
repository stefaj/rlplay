import sys
sys.path.append('/home/stefan/.local/lib/python3.6/site-packages/')
import gym
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


env = gym.make('Pendulum-v0')






# (s,o) = create_actor(sess, 2, 1, 0)



# obs = env.reset()
# while True:
#     env.render()
#     action = env.action_space.sample()
#     obs,reward,done,info= env.step(action)
#     if done: break
