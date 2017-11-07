import numpy as np
import tensorflow as tf
import gym
from replay_memory import Memory

env = gym.make('CartPole-v0')
num_actions = 2
num_obs = 4
gamma = 0.9
batch_size=128

memory = Memory(100000)

state_inp = tf.placeholder('float', [None, num_obs],name='state_inp')
policy_l1 = tf.layers.dense(inputs=state_inp, units=50, activation=tf.nn.relu)
policy_out = tf.layers.dense(inputs=policy_l1, units=num_actions, activation=tf.nn.softmax)

act_inp = tf.placeholder('float',shape=[None, num_actions], name='act_inp')
sel_act_probs = tf.log( tf.add(tf.multiply(act_inp, policy_out),1e-6) )

policy_vars = tf.trainable_variables()
policy_var_len = len(policy_vars)

critic_targ = tf.placeholder('float', [None, 1], name='critic_qtarget')
critic_paramact = tf.layers.dense(inputs=act_inp, units=num_obs, activation=tf.nn.relu)
critic_mul = tf.layers.dense(inputs=critic_paramact*state_inp, units=100, activation=tf.nn.relu)
critic_l1 = tf.layers.dense(inputs=critic_mul, units=50, activation=tf.nn.relu)
critic_out = tf.layers.dense(inputs=critic_l1, units=1, activation=None)


cur_vars = tf.trainable_variables()
critic_vars = cur_vars[policy_var_len:]
critic_var_len = len(critic_vars)

critic_grads = tf.gradients(critic_out, act_inp)
critic_loss = tf.nn.l2_loss(critic_targ - critic_out)
critic_opt = tf.train.AdamOptimizer(0.001).minimize(critic_loss)

policy_crit_grad = tf.placeholder('float', [None, num_actions], name='policy_crit_grad') 
policy_grads = tf.gradients(policy_out, policy_vars)
grads = tf.gradients(policy_out,policy_vars,-policy_crit_grad)

policy_opt_ = tf.train.AdamOptimizer(0.001)
apply_grads = policy_opt_.apply_gradients(grads_and_vars=zip(grads,policy_vars))


sess = tf.Session()
sess.run(tf.global_variables_initializer())



def rollout():
    obs = env.reset()
    transitions = []
    total_reward = 0
    while True:
        a = action_policy(obs)
        new_obs, reward, done, info = env.step(a)
        total_reward += reward
        transitions.append((obs,a,reward,(),done)) # s a r s' d
        obs = new_obs
        if done: break
    return (total_reward, transitions)

def discount_rewards(rewards):
    rs = np.zeros(len(rewards))
    g = 0
    for i in reversed(xrange(0,len(rewards))):
        g = g*gamma + rewards[i]
        rs[i] = g
    return rs

def get_advantage(actions, states,rewards):
    qs = sess.run(critic_out, feed_dict={state_inp:states, act_inp:actions })
    rs = []
    for (q,r) in zip(qs,rewards): 
        rs.append([r-q[0]])
    return rs

def action_policy(state):
    probs = sess.run(policy_out, feed_dict={state_inp:[state]})
    # return np.random.choice(num_actions, None, p=probs[0])
    a1 = 0 if np.random.uniform() < probs[0][0] else 1
    a2 = 0 if np.random.uniform() < 0.5 else 1
    return a2 if np.random.uniform() < 0.01 else a1
    # return a1

rr = 0.0
for i in range(0,10000):

    obs = env.reset()
    total_reward = 0
    while True:
        act = action_policy(obs)
        old_obs = obs
        obs,reward,done,info = env.step(act)
        act_vec = np.zeros(num_actions)
        act_vec[act] = 1
        total_reward += reward
        memory.append( (old_obs, act_vec, reward, obs, done) )  # sars'
        if done:
            break
        # update 
        (batch_states,batch_actions,batch_rewards, batch_new_states, batch_dones) \
                = memory.sample_unpack(batch_size=32)
        q_future = sess.run(critic_out, feed_dict={state_inp: batch_new_states, act_inp: batch_actions})
        q_now = []
        for j in range(0,len(batch_states)):
            t = batch_rewards[j]
            if not batch_dones[j]: t+= gamma*q_future[j][0]
            q_now.append([t])
        sess.run(critic_opt, feed_dict={state_inp:batch_states, act_inp:batch_actions, critic_targ:q_now})
        pred_acts = sess.run(policy_out, feed_dict={state_inp: batch_states})
        cgrads = sess.run(critic_grads, feed_dict={state_inp: batch_states, act_inp:pred_acts})
        sess.run(apply_grads, feed_dict={state_inp: batch_states, policy_crit_grad: cgrads[0]})
    
    rr = 0.99*rr + 0.01*total_reward
    print( "Episode %d Rolling Reward %d" % (i,rr) )

raw_input("PresS the any KeY")

obs = env.reset()
while True:
    env.render()
    probs = sess.run(policy_out, feed_dict={state_inp:[obs]})
    a = np.random.choice(num_actions, None, p=probs[0])
    obs,rewards,done,info = env.step(a)
    if done: break
