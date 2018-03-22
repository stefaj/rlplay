import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')

num_obs = 4
num_act = 2
gamma = 0.99
batch_size=64
epsilon = 0.9

mem = []


state_inp = tf.placeholder('float32', shape=(None, num_obs), name='state_inp')
l1 = tf.layers.dense(state_inp, 100, activation=tf.nn.relu)
l1 = tf.layers.dense(l1, 100, activation=tf.nn.relu)
outs = tf.layers.dense(l1, num_act, activation=None)
best_q = tf.reduce_max(outs, axis=1)
best_a = tf.argmax(outs, axis=1)

q_targ = tf.placeholder('float32', shape=(None, 1), name='q_targ')
act_inp = tf.placeholder('int32', shape=(None, ), name='act_inp')
masked = tf.gather(outs, indices=act_inp, axis=1)
loss = tf.reduce_mean( tf.nn.l2_loss(masked - q_targ) )

optim = tf.train.AdamOptimizer(0.00001).minimize(loss)


sess = tf.Session()
sess.run( tf.global_variables_initializer() )


def sample_from_mem():
    indices = np.random.choice( len(mem), batch_size)
    samples = []
    for i in indices:
        samples.append( mem[i] )
    return samples

def add_to_mem(ds):
    global mem
    for d in ds: mem.append(d)
    mem = mem[-1000000:]


def action_policy(env, obs):
    if np.random.uniform() < epsilon:
        return env.action_space.sample()
    else:
        a = sess.run(best_a, feed_dict={state_inp: [obs]} )[0]
        return a

def run_episode(env):
    transitions = []

    done = False
    obs = env.reset()
    while not done:
        action = action_policy(env, obs)
        new_obs, reward, done, info = env.step(action)

        # act_vec = np.zeros(num_act)
        # act_vec[action] = 1.0
        transitions.append( (obs, action, reward, new_obs, done) )
        obs = new_obs

    return transitions

def calc_q_targs(rewards, new_states, dones):
    q_next = sess.run( best_q, feed_dict={state_inp: new_states} )
    q_targs = rewards + (1-dones)*gamma*q_next
    return np.reshape(q_targs, (-1, 1))

def train():
    global epsilon
    hist_rew = 0.0
    for epoch in range(0,1000000):
        trans = run_episode(env)
        total_rew = np.sum( [ t[2] for t in trans ] )
        hist_rew = 0.9*hist_rew + 0.1*total_rew

        add_to_mem(trans)
        
        losses = []
        for b in range(0,100):
            batches = sample_from_mem()

            cur_states = [ t[0] for t in batches ]
            acts = [ t[1] for t in batches ]
            rewards = [ t[2] for t in batches ]
            new_states = [ t[3] for t in batches ]
            dones = [ t[4] for t in batches ]

            targets = calc_q_targs(np.array(rewards), np.array(new_states), np.array(dones))
            (l,_) = sess.run( [loss, optim], feed_dict={ state_inp: cur_states
                ,q_targ: targets, act_inp: acts} )
            losses.append(l)

        if epoch % 10 == 0: 
            print("Loss %f" % np.mean(losses))
            print("Epsilon %f" % epsilon)
            print("Reward %f" % hist_rew)
            print("Memory %d" % len(mem))
            if epsilon > 0.01: epsilon *= 0.99


train()
