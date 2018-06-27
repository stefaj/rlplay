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
N = 1
batch_size=32

def quantile_loss(a,b,tau):
    print(b)
    u = (a-b)
    delta = tf.cast(u < 0, 'float32')
    return tf.abs(tau - delta) * tf.losses.huber_loss(a,b)

state_inp = tf.placeholder('float32', shape=(None, num_obs), name='state_inp')
quantile_midpoints_inp = tf.placeholder('float32', [None, N], name="quantile_midpoints")
action_inp= tf.placeholder('float32', name='action_input', shape=(None,num_act) )
sampled_critic_gradients = tf.placeholder('float32',[None, num_act], name='sampled_critic_grads')
quantiles = tf.placeholder('float32', shape=(None, N), name="quantiles")
target = quantiles

actor_l1 = tf.layers.dense(inputs=state_inp, units=100, activation=tf.nn.relu)
actor_l1 = tf.layers.dense(inputs=actor_l1, units=100, activation=tf.nn.relu)
actor_out = tf.layers.dense(inputs=actor_l1, units=num_act, activation=tf.nn.softmax)
actor_weights = tf.trainable_variables()


critic_state = tf.layers.dense(inputs=state_inp, units=100, activation=tf.nn.relu)
critic_act = tf.layers.dense(inputs=action_inp, units=100, activation=tf.nn.relu)
critic_l1 = tf.layers.dense(inputs=tf.concat([critic_state, critic_act],axis=1), units=100
        , activation=tf.nn.relu)

net = tf.layers.dense(critic_l1, units=100, activation=tf.nn.relu)
net = tf.layers.dense(net, units=100, activation=tf.nn.relu)

critic_output = quantiles_locations = tf.layers.dense(net, N)
critic_q_output = tf.reduce_sum(1.0 / float(N) * critic_output, axis=1)

critic_weights = tf.trainable_variables()[ len(actor_weights): ]
critic_action_grads = tf.gradients(critic_q_output, action_inp)

actor_params_grad = tf.gradients(ys=actor_out, xs=actor_weights, grad_ys=sampled_critic_gradients)
actor_grads = zip(actor_params_grad,actor_weights)
actor_opt_ = tf.train.AdamOptimizer(-1e-5)
apply_grads = actor_opt_.apply_gradients(grads_and_vars=actor_grads)



theta_i = tf.tile(tf.expand_dims(quantiles_locations, -1), [1, 1, N])
T_theta_j = tf.tile(tf.expand_dims(target, -2), [1, N, 1])
tau_i = tf.tile(tf.expand_dims(quantile_midpoints_inp, -1), [1, 1, N])

kappa = 1.0
error = T_theta_j - theta_i
abs_error = tf.abs(error)
quadratic = tf.minimum(abs_error, kappa)
huber_loss = kappa * (abs_error - quadratic) + 0.5 * quadratic ** 2
quantile_huber_loss = tf.abs(tau_i - tf.cast(error < 0, dtype=tf.float32)) * huber_loss
quantile_regression_loss = tf.reduce_sum(quantile_huber_loss) / float(N)







optim = tf.train.AdamOptimizer(1e-2).minimize(quantile_regression_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mem = []


q_probs = np.ones(N) / float(N)


def action_policy(env, state):
    probs = sess.run(actor_out, feed_dict={state_inp:[state]})
    if np.random.uniform() < 0.5:
        return env.action_space.sample()
    return np.random.choice(num_act, p=probs[0])


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


        next_act = sess.run(actor_out, feed_dict={state_inp: ss_})
        calc_cur_thetas = sess.run(critic_output, feed_dict={state_inp: ss, action_inp: as_} )
        calc_next_thetas = sess.run(critic_output, feed_dict={state_inp: ss_, action_inp: next_act} )

        batch_idx = list(range(batch_size))

        cumulative_probabilities = np.array(range(N+1))/float(N)  # tau_i
        quantile_midpoints = 0.5*(cumulative_probabilities[1:] + cumulative_probabilities[:-1])  # tau^hat_i
        quantile_midpoints = np.tile(quantile_midpoints, (batch_size, 1))
        # sorted_quantiles = np.argsort(calc_cur_thetas[batch_idx, np.argmax(as_)])
        # for idx in range(batch_size):
        #     quantile_midpoints[idx, :] = quantile_midpoints[idx, sorted_quantiles[idx]]

        targets = rs + (1.0 - ds)*gamma*calc_next_thetas

        # print(targets.shape)
        assert targets.shape == (batch_size, N)

        (l, _) = sess.run([quantile_regression_loss, optim], feed_dict={target: targets, state_inp: ss, 
            quantile_midpoints_inp:quantile_midpoints, action_inp: as_})

        cur_act = sess.run(actor_out, feed_dict={state_inp: ss})
        act_for_grads = sess.run(critic_action_grads, feed_dict={state_inp:ss, action_inp:cur_act})[0]
        sess.run(apply_grads, feed_dict={state_inp: ss, sampled_critic_gradients:act_for_grads} )

    if epoch % 100 == 0:
        print("total reward %f " % np.mean(past_rews) )
        print("Loss %f" % l)



(s, a, r, s_, d) = run_episode(env)




