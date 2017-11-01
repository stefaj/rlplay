import gym
import numpy as np
import tensorflow as tf
from ReplayBuffer import ReplayBuffer

def create_actor(sess,n_features,n_actions,action_range):
    state = tf.placeholder('float',[None, n_features])
    layer = tf.layers.dense(inputs=state, units=100, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=layer, units=n_actions, activation=None)
    output = tf.multiply( tf.tanh(output) , action_range)
    return (state,output)

def create_critic(sess,n_features,n_actions):
    state = tf.placeholder('float', [None, n_features])
    action = tf.placeholder('float', [None, n_actions])
    state_layer = tf.layers.dense(inputs=state, units=100, activation=tf.nn.relu)
    action_layer = tf.layers.dense(inputs=action, units=100, activation=tf.nn.relu)
    comb_layer = tf.concat([state_layer,action_layer],1)
    out = tf.layers.dense(inputs=comb_layer, units=1, activation=tf.nn.softmax) # activ might be None?
    return state, action, out



env = gym.make('Pendulum-v0')






# (s,o) = create_actor(sess, 2, 1, 0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())




# obs = env.reset()
# while True:
#     env.render()
#     action = env.action_space.sample()
#     obs,reward,done,info= env.step(action)
#     if done: break
