import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
num_actions = 2
num_obs = 4
gamma = 0.9

state_inp = tf.placeholder('float', [None, num_obs])
actor_l1 = tf.layers.dense(inputs=state_inp, units=50, activation=tf.nn.relu)
for _ in range(0,10):
    actor_l1 = tf.layers.dense(inputs=actor_l1, units=50, activation=tf.nn.relu)
actor_out = tf.layers.dense(inputs=actor_l1, units=num_actions, activation=tf.nn.softmax)
# actor_qhat = tf.placeholder('float',[None, 1], name='actor_qhat')
actor_weights = tf.trainable_variables()

act_inp = tf.placeholder('float',shape=[None, num_actions], name='act_inp')
sel_act_probs = tf.add(tf.multiply(act_inp, actor_out),1e-6)


critic_l1 = tf.layers.dense(inputs=state_inp, units=50, activation=tf.nn.relu)
critic_l2 = tf.layers.dense(inputs=tf.concat([critic_l1, act_inp],1), units=100, activation=tf.nn.relu)
for _ in range(0,1):
    critic_l2 = tf.layers.dense(inputs=critic_l2, units=50, activation=tf.nn.relu)
critic_out = tf.layers.dense(inputs=critic_l2, units=1, activation=None)
critic_weights = tf.trainable_variables()[ len(actor_weights): ]

critic_action_grads = tf.gradients(critic_out, act_inp)

critic_rewards = tf.placeholder('float32')
critic_targ = critic_rewards + gamma*critic_out
# critic_loss = tf.nn.l2_loss(critic_targ - critic_out)
critic_loss = tf.reduce_mean(tf.squared_difference(critic_targ, critic_out))
critic_opt = tf.train.AdamOptimizer(0.01).minimize(critic_loss)

sampled_critic_gradients = tf.placeholder('float32',[None, num_actions])
actor_params_grad = tf.gradients(ys=actor_out, xs=actor_weights, grad_ys=critic_action_grads)
actor_grads = zip(actor_params_grad,actor_weights)
actor_opt_ = tf.train.AdamOptimizer(-0.01)
apply_grads = actor_opt_.apply_gradients(grads_and_vars=actor_grads)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

def action_policy(state):
    probs = sess.run(actor_out, feed_dict={state_inp:[state]})
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
        transitions.append((obs,a,reward,new_obs,done)) # s a r s' d
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

def get_critic_q(states, actions):
    qs = sess.run(critic_out, feed_dict={state_inp:states, act_inp: actions})
    return qs

rr = 0.0
for i in range(0,10000):
    (tr, transitions) = rollout()
    rr = 0.99*rr + 0.01*tr

    (states, rewards, new_states, actions) = ([],[],[],[])
    rewards = []
    # print(targets.shape)

    for j in range(0,len(transitions)):
        states.append(transitions[j][0])
        new_states.append(transitions[j][3])
        act_vec = np.zeros(num_actions)
        act_vec[transitions[j][1]] = 1
        actions.append(act_vec)  
        rewards.append(transitions[j][2])

    # print('states',states)
    # print('actions',actions)
    # print('target',rewards)
    # print(len(rewards))

    act_for_grads = sess.run(actor_out, feed_dict={state_inp:states})
    sess.run(apply_grads, feed_dict={state_inp: states, act_inp:act_for_grads} )

    sess.run(critic_opt, feed_dict={state_inp: states, critic_rewards:rewards, act_inp: actions} )

    if i % 10 == 0: print("Iteration %d Rolloing Reward: %d" % (i,rr))

raw_input("PresS the any KeY")

obs = env.reset()
while True:
    env.render()
    probs = sess.run(actor_out, feed_dict={state_inp:[obs]})
    a = np.random.choice(num_actions, None, p=probs[0])
    obs,rewards,done,info = env.step(a)
    if done: break
