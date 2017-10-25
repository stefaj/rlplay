import gym
import numpy as np
import random
import tensorflow as tf

env = gym.make('CartPole-v0')
env.reset()
action = env.action_space.sample()
params = np.random.rand(4)*2-1


def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("pol_params", [4,2])
        state = tf.placeholder("float", [None,4])
        actions = tf.placeholder("float", [None,2])
        advantages = tf.placeholder("float", [None,1])
        linear = tf.matmul(state,params)
        probs = tf.nn.softmax(linear)
        good_probs = tf.reduce_sum(tf.multiply(probs, actions), reduction_indices=[1])
        elig = tf.log(good_probs) * advantages
        loss = -tf.reduce_sum(elig)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probs, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float", [None,4])
        w1 = tf.get_variable("w1", [4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2", [10,1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1,w2) + b2

        newvals = tf.placeholder("float", [None,1])
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    obs = env.reset()
    total = 0
    (states,actions,advantages,transitions,update_vals) = ([],[],[],[],[])

    for i in range(0,500):
        obs_vector = np.expand_dims(obs, axis=0) 
        probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        states.append(obs)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        old_obs = obs
        obs, reward, done, info = env.step(action)
        transitions.append((old_obs, action, reward))
        total += reward
        if done: break

    for index, trans in enumerate(transitions):
        obs_, action, reward = trans
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease  = 1
        for index2 in range(0,future_transitions):
            future_reward += transitions[index2 + index][2] * decrease
            decrease *= 0.97
        obs_vector = np.expand_dims(obs_, axis=0)
        current_val = sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]
        advantages.append(future_reward - current_val)
        update_vals.append(future_reward)

    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector,
        pl_actions: actions})

    return total

policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(0,5000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    if i % 100 == 0: print("iteration %d" % i)

pl_calculated, pl_state, _, _, _ = policy_grad
obs = env.reset()

for i in range(0,1000):
   env.render()

   obs_vector = np.expand_dims(obs, axis=0) 
   probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})
   action = 0 if random.uniform(0,1) < probs[0][0] else 1
   obs, reward, done, info = env.step(action)
