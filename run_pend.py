"""
Dueling DQN & Natural DQN comparison

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DuelingDQN
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('CartPole-v0')
MEMORY_SIZE = 100

sess = tf.Session()
with tf.variable_scope('natural'):
    natural_DQN = DuelingDQN(
        n_actions=2, n_features=4, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.0001, sess=sess, dueling=False)

sess.run(tf.global_variables_initializer())


def train(RL):
    acc_r = [0]
    total_steps = 0
    observation = env.reset()

    for i in range(0,10000):
        observation = env.reset()
        total = 0
        while True:
            # if total_steps-MEMORY_SIZE > 9000: env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            total += reward

            acc_r.append(reward/10.0 + acc_r[-1])  

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if total_steps % 100 == 0: print("episode %d reward %d" % (i,total) )

            if done: break

            observation = observation_
            total_steps += 1

# train(dueling_DQN)
train(natural_DQN)
