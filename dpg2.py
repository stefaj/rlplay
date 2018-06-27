import numpy as np
import tensorflow as tf
import gym
import L4

env = gym.make('CartPole-v0')
num_actions = 2
num_obs = 4
gamma = 0.9

state_inp = tf.placeholder('float', [None, num_obs])
act_inp = tf.placeholder('float',shape=[None, num_actions], name='act_inp')

actor_l1 = tf.layers.dense(inputs=state_inp, units=10, activation=tf.nn.relu)
for _ in range(10):
    actor_l1 = tf.layers.dense(inputs=actor_l1, units=10, activation=tf.nn.relu)

actor_out = tf.layers.dense(inputs=actor_l1, units=num_actions, activation=tf.nn.softmax)
actor_weights = tf.trainable_variables()

critic_state = tf.layers.dense(inputs=state_inp, units=20, activation=tf.nn.relu)
critic_act = tf.layers.dense(inputs=act_inp, units=20, activation=tf.nn.relu)
critic_l1 = tf.layers.dense(inputs=tf.concat([critic_state, critic_act],axis=1), units=20
        , activation=tf.nn.relu)
for _ in range(10):
    critic_l1 = tf.layers.dense(inputs=critic_l1, units=20, activation=tf.nn.relu)

critic_out = tf.layers.dense(inputs=critic_l1, units=1, activation=None)
critic_weights = tf.trainable_variables()[ len(actor_weights): ]

critic_action_grads = tf.gradients(critic_out, act_inp)

critic_targ = tf.placeholder('float32', shape=[None, 1] )
# critic_loss = tf.nn.l2_loss(critic_targ - critic_out)
critic_loss = tf.reduce_mean(tf.squared_difference(critic_targ, critic_out))
critic_opt = tf.train.AdamOptimizer(0.01).minimize(critic_loss)

sampled_critic_gradients = tf.placeholder('float32',[None, num_actions], name='sampled_critic_grads')
actor_params_grad = tf.gradients(ys=actor_out, xs=actor_weights, grad_ys=sampled_critic_gradients)
actor_grads = zip(actor_params_grad,actor_weights)
actor_opt_ = tf.train.AdamOptimizer(-1e-4)
apply_grads = actor_opt_.apply_gradients(grads_and_vars=actor_grads)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

def action_policy(state):
    probs = sess.run(actor_out, feed_dict={state_inp:[state]})
    # return np.random.choice(num_actions, None, p=probs[0])
    if np.random.uniform() < 0.05:
        return env.action_space.sample()
    return np.random.choice(num_actions, p=probs[0])

def rollout():
    obs = env.reset()
    transitions = []
    while True:
        a = action_policy(obs)
        new_obs, reward, done, info = env.step(a)
        transitions.append((obs,a,reward,new_obs,done)) # s a r s' d
        obs = new_obs
        if done: break
    return transitions

def discount_rewards(rewards):
    rs = np.zeros(len(rewards))
    g = 0
    for i in reversed(range(0,len(rewards))):
        g = g*gamma + rewards[i]
        rs[i] = g
    return rs

def get_critic_q(states, actions):
    qs = sess.run(critic_out, feed_dict={state_inp:states, act_inp: actions})
    return qs

mem = []

rr = 0.0
<<<<<<< HEAD
for i in range(0,1000):
    (tr, transitions) = rollout()
    rr = 0.99*rr + 0.01*tr

    (states, rewards, actions) = ([],[],[])
    rs = []
    rewards = get_advantage( [ t[0] for t in transitions ], [ t[2] for t in transitions] )
    for j in range(0,len(transitions)):
        states.append(transitions[j][0])
        act_vec = np.zeros(num_actions)
        act_vec[transitions[j][1]] = 1
        actions.append(act_vec)  
        rs.append(transitions[j][2])
    
    # print('states',states)
    # print('actions',actions)
    # print('target',rewards)
    sess.run(apply_grads, feed_dict={state_inp: states, act_inp: actions, policy_qhat:rewards} )
    targ = [ [r] for r in discount_rewards(rs) ]
    sess.run(critic_opt, feed_dict={state_inp: states, critic_targ:targ } )
# =======
# for i in range(0,10000):
#     transitions = rollout()
#     rr = 0.99*rr + 0.01*np.sum([ s[2] for s in transitions ])
#     mem = (mem + transitions)[-100000:]
# 
#     if len(mem) < 10000: continue
# 
#     for b in range(3):
#         indices = np.random.choice( len(mem), 128)
#         samples = [ mem[idx] for idx in indices ]
# 
#         (states, rewards, new_states, actions, dones) = ([],[],[],[], [])
#         rewards = []
#         # print(targets.shape)
# 
#         for j in range(0,len(samples)):
#             states.append(samples[j][0])
#             new_states.append(samples[j][3])
#             act_vec = np.zeros(num_actions)
#             act_vec[samples[j][1]] = 1
#             actions.append(act_vec)  
#             rewards.append(samples[j][2])
#             dones.append(samples[j][4])
# 
#         next_act = sess.run(actor_out, feed_dict={state_inp: new_states})
#         q_next = sess.run(critic_out, feed_dict={state_inp: new_states, act_inp: next_act})
#         rewards = np.array(rewards).reshape( (-1,1) )
#         dones = np.array(dones).reshape( (-1,1) )
#         q_targ = rewards + gamma*(1-dones)*q_next
# 
#         sess.run(critic_opt, feed_dict={state_inp: states, critic_targ: q_targ, act_inp: actions})
# 
#         cur_act = sess.run(actor_out, feed_dict={state_inp: states})
#         act_for_grads = sess.run(critic_action_grads, feed_dict={state_inp:states, act_inp:cur_act})[0]
#         sess.run(apply_grads, feed_dict={state_inp: states, sampled_critic_gradients:act_for_grads} )
# # >>>>>>> 17c2e37a4c170bbfb68552039049b278dc0bcd60

    if i % 10 == 0: print("Iteration %d Rolloing Reward: %d" % (i,rr))

raw_input("PresS the any KeY")

obs = env.reset()
while True:
    env.render()
    probs = sess.run(actor_out, feed_dict={state_inp:[obs]})
    a = np.random.choice(num_actions, None, p=probs[0])
    obs,rewards,done,info = env.step(a)
    if done: break
