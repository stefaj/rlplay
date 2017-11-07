import numpy as np
import tensorflow as tf
import gym
from replay_memory import Memory

epsilon = 0.1
gamma = 0.9
num_atoms=51

env = gym.make('CartPole-v0')
num_actions=2
num_obs=4

batch_size=64

state_inp = tf.placeholder('float', [None, num_obs])
layer1 = tf.layers.dense(inputs=state_inp, units=50, activation=tf.nn.relu)
layer2 = tf.layers.dense(inputs=layer1, units=20, activation=tf.nn.relu)

dists = []
for i in range(0,num_actions):
    dists.append(tf.layers.dense(inputs=layer2, units=num_atoms, activation=tf.nn.softmax))

m_inp = tf.placeholder('float', [num_actions,None,num_atoms])
loss = -tf.reduce_sum( m_inp * tf.log(tf.add(dists,1e-6)) )

optim = tf.train.AdamOptimizer(0.0001).minimize(loss)

v_min = 0.0
v_max = 100.0
z_delta = (v_max-v_min)/float(num_atoms-1)
z_hist = [v_min + z_delta*float(i) for i in range(0,num_atoms)]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

memory = Memory(100000)

rr = 0
for i in range(0,2000):
    obs = env.reset()
    total = 0
    while True:

        a = 0
        if np.random.uniform() < epsilon:
            a = env.action_space.sample()
        else:
            zs = sess.run(dists, feed_dict={state_inp: [obs]} )
            z_concat = np.vstack(zs)
            qs = np.multiply(z_concat, np.array(z_hist))
            q_expect = np.sum(qs,axis=1)
            a = np.argmax( q_expect )
        if epsilon > 0.01: epsilon *= 0.999999

        obs_new,reward,done,info =  env.step(a)
        total += reward
   
        memory.append( (obs,a,reward,obs_new,done,()) )   

        targets=[]
        states=[]
        next_states=[]
        samples = memory.sample(None,batch_size) 
        targets=[]

        next_states = [s[3] for s in samples]
        pss = sess.run(dists, feed_dict={state_inp:next_states})
        s__ = np.transpose(pss, [1,0,2])

        m_prob = [np.zeros((batch_size, num_atoms)) for _ in range(num_actions)]
        for (k,(ps, (s_,a_,r_,sp_,done_,_))) in enumerate(zip(s__,samples)):
            q_expect = np.sum(np.multiply(np.vstack(ps), np.array(z_hist)), axis=1)
            opt_a = int(np.argmax( q_expect ))
            a_ = int(a_)
            if done_: 
                tz = min(v_max, max(v_min, r_))
                bj = (tz - v_min) / z_delta 
                l, u = np.floor(bj), np.ceil(bj)
                m_prob[a_][k][int(l)] += (u - bj)
                m_prob[a_][k][int(u)] += (bj - l)
            else:
                for j in range(num_atoms):
                    tz = min(v_max, max(v_min, r_ + gamma * z_hist[j]))
                    bj = (tz - v_min) / z_delta 
                    l, u = np.floor(bj), np.ceil(bj)
                    m_prob[a_][k][int(l)] += ps[opt_a][j] * (u - bj)
                    m_prob[a_][k][int(u)] += ps[opt_a][j] * (bj - l)

            states.append(s_)

        sess.run(optim, feed_dict={m_inp: m_prob, state_inp: states})
    
        obs = obs_new
        if done: break
    rr = 0.99*rr + 0.01*total
    print("iteration %d avg rew %d epsilon %f" % (i,rr,epsilon))

raw_input("Press any key to play")
obs = env.reset()
while True:
    env.render()
    a = sess.run(out, feed_dict={state_inp: [obs]} )[0]
    a = np.argmax( a )
    obs,reward,done,info = env.step(a)
    if done: break
