import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
num_actions = 2
num_obs = 4
gamma = 0.9

state_inp = tf.placeholder('float', [None, num_obs],name='state_inp')
policy_l1 = tf.layers.dense(inputs=state_inp, units=50, activation=tf.nn.relu)
policy_out = tf.layers.dense(inputs=policy_l1, units=num_actions, activation=tf.nn.softmax)
policy_qhat = tf.placeholder('float',[None, 1],name='policy_qhat')

act_inp = tf.placeholder('float',shape=[None, num_actions])
# sel_act = tf.argmax(act_inp, output_type=tf.int32, axis=1)
# indices = tf.range(0, tf.shape(policy_out)[0]) * tf.shape(policy_out)[1] + sel_act
# sel_act_probs = tf.gather(tf.reshape(policy_out, [-1]), indices)
sel_act_probs = tf.add(tf.multiply(act_inp, policy_out),1e-6)


critic_actions = tf.placeholder('float', [None, num_actions])
critic_targ = tf.placeholder('float', [None, 1])
critic_prestate = tf.layers.dense(inputs=state_inp, units=21, activation=tf.nn.relu)
critic_params = tf.get_variable('critic_params', [21, 5])
critic_paramact = tf.matmul(tf.layers.dense(inputs=critic_actions, units=21, activation=tf.nn.relu)
        , critic_params)
linear = tf.matmul(critic_prestate, critic_params)
critic_l1 = tf.layers.dense(inputs=tf.concat([linear, critic_prestate],axis=1), units=10, activation=tf.nn.relu)

# critic_l2 = tf.layers.dense(inputs=state_inp, units=50, activation=tf.nn.relu)
# critic_l3 = tf.layers.dense(inputs=tf.concat([critic_l2, critic_actions],axis=1), units=20, activation=tf.nn.relu)
# critic_l4 = tf.layers.dense(inputs=critic_l3, units=5, activation=tf.nn.relu)

critic_out = tf.layers.dense(inputs=critic_l1, units=1, activation=None)
critic_loss = tf.nn.l2_loss(critic_targ - critic_out)
critic_opt = tf.train.AdamOptimizer(0.001).minimize(critic_loss)

policy_crit_grad = tf.placeholder('float', [None, num_actions]) 
policy_loss = -tf.reduce_mean( tf.multiply(policy_qhat, tf.log(sel_act_probs) ) )
policy_grads = tf.gradients(policy_loss, tf.trainable_variables())
# critic_grads = tf.gradients(critic_loss, tf.trainable_variables())
# deter_grads = tf.multiply(policy_grads, critic_grads)
policy_opt_ = tf.train.AdamOptimizer(0.01)
# policy_opt = policy_opt_.minimize(policy_loss)
apply_grads = policy_opt_.apply_gradients(grads_and_vars=zip(policy_grads,tf.trainable_variables()))


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
    qs = sess.run(critic_out, feed_dict={state_inp:states, critic_actions:actions })
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
    sess.run(apply_grads, feed_dict={state_inp: states, act_inp: actions, policy_qhat:rewards} )
    targ = [ [r] for r in discount_rewards(rs) ]
    sess.run(critic_opt, feed_dict={state_inp: states, critic_targ:targ, critic_actions:actions} )

    if i % 10 == 0: print("Iteration %d Rolloing Reward: %d" % (i,rr))

raw_input("PresS the any KeY")

obs = env.reset()
while True:
    env.render()
    probs = sess.run(policy_out, feed_dict={state_inp:[obs]})
    a = np.random.choice(num_actions, None, p=probs[0])
    obs,rewards,done,info = env.step(a)
    if done: break
