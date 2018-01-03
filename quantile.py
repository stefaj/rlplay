import numpy as np
import tensorflow as tf
import gym
from collections import deque

def gather_along_second_axis(data, indices):
    batch_offset = tf.range(0, tf.shape(data)[0])
    flat_indices = tf.stack([batch_offset, indices], axis=1)
    return tf.gather_nd(data, flat_indices)

env = gym.make('CartPole-v0')

num_obs = 4
num_act = 2
gamma = 0.9
kappa = 1
N = 50
batch_size=32

def quantile_loss(a,b,tau):
    print(b)
    u = (a-b)
    delta = tf.cast(u < 0, 'float32')
    return tf.abs(tau - delta) * tf.losses.huber_loss(a,b)

state_inp = tf.placeholder('float32', shape=(None, num_obs), name='state_inp')
act_inp= tf.placeholder('int32', name='act_inp', shape=(None,2) )
quantile_midpoints_inp = tf.placeholder(tf.float32, [None, N], name="quantile_midpoints")


net = tf.layers.dense(state_inp, units=400, activation=tf.nn.relu)
net = tf.layers.dense(net, units=200, activation=tf.nn.relu)
net = tf.layers.dense(net, units=200, activation=tf.nn.relu)

quantiles_locations = tf.layers.dense(net, num_act * N)
quantiles_locations = tf.reshape(quantiles_locations, (tf.shape(quantiles_locations)[0], num_act, N), 'quantiles_locations')
output = quantiles_locations
print(output)

quantiles = tf.placeholder(tf.float32, shape=(None, N), name="quantiles")
target = quantiles
quantiles_for_used_actions = tf.gather_nd(quantiles_locations, act_inp)

theta_i = tf.tile(tf.expand_dims(quantiles_for_used_actions, -1), [1, 1, N])
T_theta_j = tf.tile(tf.expand_dims(target, -2), [1, N, 1])
tau_i = tf.tile(tf.expand_dims(quantile_midpoints_inp, -1), [1, 1, N])

kappa = 1.0
error = T_theta_j - theta_i
abs_error = tf.abs(error)
quadratic = tf.minimum(abs_error, kappa)
huber_loss = kappa * (abs_error - quadratic) + 0.5 * quadratic ** 2
quantile_huber_loss = tf.abs(tau_i - tf.cast(error < 0, dtype=tf.float32)) * huber_loss
quantile_regression_loss = tf.reduce_sum(quantile_huber_loss) / float(N)

optim = tf.train.AdamOptimizer(1e-4).minimize(quantile_regression_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mem = []


q_probs = np.ones(N) / float(N)

def action_policy(env, obs):
    if np.random.uniform() < 0.1:
        return env.action_space.sample()
    else:
        calc_next_thetas = sess.run(output, feed_dict={state_inp: [obs]} )[0]

        next_qs = np.dot( calc_next_thetas, q_probs )
        target_actions = np.argmax(next_qs)

        return target_actions


def run_episode(env):
    transitions = []

    done = False
    obs = env.reset()
    while not done:
        action = action_policy(env, obs)
        new_obs, reward, done, info = env.step(action)

        act_vec = np.zeros(num_act)
        act_vec[action] = 1.0
        transitions.append( (obs, act_vec, reward, new_obs, done) )
        obs = new_obs

    return transitions


[past_rews] = [deque(maxlen=100)]
for epoch in range(0, 100000):
    vals = run_episode(env)
    mem = mem + vals
    mem = mem[-100000:]
    past_rews.append( np.sum([ v[2] for v in vals] ) )
    
    for _ in range(0,2): # batches
        indices = np.random.choice( len(mem), batch_size)
        samples = []
        for i in indices:
            samples.append( mem[i] )

        ss = np.array([ s[0] for s in samples ]).reshape( (batch_size, num_obs) )
        as_ = np.array([ s[1] for s in samples ]).reshape( (batch_size, num_act) )
        rs = np.array([ s[2] for s in samples ]).reshape( (batch_size,1) )
        ss_ = np.array([ s[3] for s in samples ]).reshape( (batch_size,num_obs) )
        ds = np.array([ s[4] for s in samples ]).reshape( (batch_size,1) )

        calc_cur_thetas = sess.run(output, feed_dict={state_inp: ss} )
        calc_next_thetas = sess.run(output, feed_dict={state_inp: ss_} )

        next_qs = np.dot( calc_next_thetas, q_probs )
        target_actions = np.argmax(next_qs, axis=1)
    
        calc_q_targ = []
        batch_idx = list(range(batch_size))

        acts = [[b, np.argmax(a)] for b, a in zip(batch_idx, as_)]
        # print(acts)

        cumulative_probabilities = np.array(range(N+1))/float(N)  # tau_i
        quantile_midpoints = 0.5*(cumulative_probabilities[1:] + cumulative_probabilities[:-1])  # tau^hat_i
        quantile_midpoints = np.tile(quantile_midpoints, (batch_size, 1))
        sorted_quantiles = np.argsort(calc_cur_thetas[batch_idx, np.argmax(as_)])
        for idx in range(batch_size):
            quantile_midpoints[idx, :] = quantile_midpoints[idx, sorted_quantiles[idx]]

        targets = rs + (1.0 - ds)*gamma*calc_next_thetas[batch_idx, target_actions]

        # print(targets.shape)
        assert targets.shape == (batch_size, N)

        (l, _) = sess.run([quantile_regression_loss, optim], feed_dict={target: targets, state_inp: ss, act_inp: acts, quantile_midpoints_inp:quantile_midpoints})

    if epoch % 100 == 0:
        print("total reward %f " % np.mean(past_rews) )
        print("Loss %f" % l)



(s, a, r, s_, d) = run_episode(env)




