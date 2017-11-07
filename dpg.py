import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
num_actions = 2
num_obs = 4
gamma = 0.9

policy_qhat = tf.placeholder('float',[None, 1],name='policy_qhat')
state_inp = tf.placeholder('float', [None, num_obs],name='state_inp')
policy_l1 = tf.layers.dense(inputs=state_inp, units=50, activation=tf.nn.relu)
policy_out = tf.layers.dense(inputs=policy_l1, units=num_actions, activation=tf.nn.softmax)

act_inp = tf.placeholder('float',shape=[None, num_actions])
sel_act_probs = tf.log( tf.add(tf.multiply(act_inp, policy_out),1e-6) )

policy_vars = tf.trainable_variables()
policy_var_len = len(policy_vars)

cstate_inp = tf.placeholder('float', [None, num_obs],name='cstate_inp')
critic_actions = tf.placeholder('float', [None, num_actions])
critic_targ = tf.placeholder('float', [None, 1])
critic_paramact = tf.layers.dense(inputs=critic_actions, units=num_obs, activation=tf.nn.relu)
linear = tf.layers.dense(inputs=critic_paramact*cstate_inp, units=100, activation=tf.nn.relu)
critic_l1 = tf.layers.dense(inputs=linear, units=50, activation=tf.nn.relu)
critic_out = tf.layers.dense(inputs=critic_l1, units=1, activation=None)


cur_vars = tf.trainable_variables()
critic_vars = cur_vars[policy_var_len:]
critic_var_len = len(critic_vars)

critic_grads = tf.gradients(critic_l1, critic_actions)
# print(critic_grads)
# raise 'poes'
critic_loss = tf.nn.l2_loss(critic_targ - critic_out)
critic_opt = tf.train.AdamOptimizer(0.0010).minimize(critic_loss)

policy_crit_grad = tf.placeholder('float', [None, num_actions]) 
policy_loss = -tf.reduce_mean( tf.multiply(policy_qhat, sel_act_probs ) )
policy_grads = tf.gradients(policy_loss, policy_vars)
grads = tf.gradients(policy_out,policy_vars )

policy_opt_ = tf.train.AdamOptimizer(0.01)
apply_grads = policy_opt_.apply_gradients(grads_and_vars=zip(policy_grads,policy_vars))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

def action_policy(state):
    probs = sess.run(policy_out, feed_dict={state_inp:[state]})
    # return np.random.choice(num_actions, None, p=probs[0])
    return 0 if np.random.uniform() < probs[0][0] else 1

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
    qs = sess.run(critic_out, feed_dict={cstate_inp:states, critic_actions:actions })
    rs = []
    for (q,r) in zip(qs,rewards): 
        rs.append([r-q[0]])
    return rs

rr = 0.0
for i in range(0,10000):
    (tr, transitions) = rollout()
    rr = 0.99*rr + 0.01*tr

    (states, rewards, actions) = ([],[],[])
    rs = []
    for j in range(0,len(transitions)):
        states.append(transitions[j][0])
        act_vec = np.zeros(num_actions)
        act_vec[transitions[j][1]] = 1
        actions.append(act_vec)  
        rs.append(transitions[j][2])

    rewards = get_advantage( actions, states, rs )
    
    # print('states',states)
    # print('actions',actions)
    # print('target',rewards)
    targ = [ [r] for r in discount_rewards(rs) ]
    sess.run(critic_opt, feed_dict={cstate_inp: states, critic_targ:targ, critic_actions:actions} )
    cgrads = sess.run(critic_grads, feed_dict={cstate_inp:states, critic_actions:actions })

    sess.run(apply_grads, feed_dict={state_inp: states, act_inp: actions, policy_qhat:rewards} )
    pgrads = sess.run(grads, feed_dict={state_inp:states})

    if i % 10 == 0: print("Iteration %d Rolloing Reward: %d" % (i,rr))

raw_input("PresS the any KeY")

obs = env.reset()
while True:
    env.render()
    probs = sess.run(policy_out, feed_dict={state_inp:[obs]})
    a = np.random.choice(num_actions, None, p=probs[0])
    obs,rewards,done,info = env.step(a)
    if done: break
