import gym
import tensorflow as tf
import numpy as np
from collections import deque

env = gym.make('CartPole-v0')
n_features = 4
n_actions = 2
lr = 0.001
critic_lr = 0.01
gamma = 0.9


sess = tf.Session()

advantage = tf.placeholder("float", [None, 1])
state_ = tf.placeholder("float", [None, n_features])
actions_ = tf.placeholder("float", [None, n_actions])
l1 = tf.layers.dense(inputs=state_, units=20, activation=tf.nn.relu
        ,kernel_initializer=tf.random_normal_initializer(0., 0.01)
        ,bias_initializer=tf.constant_initializer(0.1)) 
l2 = tf.layers.dense(inputs=l1, units=5, activation=tf.nn.relu
        ,kernel_initializer=tf.random_normal_initializer(0., 0.01)
        ,bias_initializer=tf.constant_initializer(0.1))
probs = tf.layers.dense(inputs=l2, units=n_actions, activation=tf.nn.softmax)
prob_diff = tf.log( tf.reduce_sum (tf.multiply(probs, actions_) ) ) 
loss = -tf.reduce_sum( prob_diff * advantage ) 
optim = tf.train.AdamOptimizer(lr).minimize(loss)

critic_ret = tf.placeholder('float', [None, 1])
critic_l1 = tf.layers.dense(inputs=state_, units=10, activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., 0.01))
critic_output = tf.layers.dense(inputs=critic_l1, units=1, activation=None) 
critic_loss = tf.nn.l2_loss(critic_ret - critic_output)
critic_optim = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss)


obs = env.reset()

# running_rew = 0
# while True:
#     env.render()
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     running_rew += reward
#     print("Reward %d " % running_rew)
#     if done: break


def run_episode(env, sess):
    running_rew = 0
    states = []
    actions = []
    transitions = []
    total = 0
    obs = env.reset() 
    while True:
        p = sess.run( probs, feed_dict={state_: [obs]} )
        # print('p',p)
        action = 0 if np.random.uniform() < p[0][0] else 1
        acts = np.zeros(2)
        acts[action] = 1
        actions.append(acts)
        states.append(obs)
        old_obs = obs
        obs, reward, done, info = env.step(action)
        transitions.append( (old_obs, acts, reward) )
        total += reward
        if done: break
    return (states, actions, transitions,total)

def train(sess):
    returns = deque(maxlen=100)
    rr = 0
    for i in range(1,10000):
        (states, actions, transitions,total) = run_episode(env, sess)
        returns.append(total)
        if rr == 0: rr = total
        rr = 0.99*rr + 0.01*total

        disc_rews = []
        advs = []
        for j in range(0, len(transitions)):
            (obs,acts,rew) = transitions[j]
            future_rew = 0
            g = 1.0
            for k in range(j,len(transitions)):
                future_rew += g*transitions[k][2]
                g = g * gamma
            disc_rews.append([future_rew])
            b = sess.run( critic_output, feed_dict={state_:[obs] })
            advs.append([future_rew - b[0][0]])

        ## print(advs)

        sess.run(optim, feed_dict={advantage: advs, state_: states, actions_: actions} )
        sess.run(critic_optim, feed_dict={critic_ret: disc_rews
            ,state_: states} )

        print("it %d - running reward: %d: " % (i, rr) )

sess.run(tf.global_variables_initializer())

train(sess)
