import numpy as np
import tensorflow as tf
import gym
from replay_memory import Memory

epsilon = 0.1
gamma = 0.9
num_atoms=11

env = gym.make('CartPole-v0')
num_actions=2
num_obs=4

batch_size=64


state_inp = tf.placeholder('float', [None, num_obs])
layer1 = tf.layers.dense(inputs=state_inp, units=50, activation=tf.nn.relu)
layer2 = tf.layers.dense(inputs=layer1, units=20, activation=tf.nn.relu)

# out = tf.layers.dense(inputs=layer4, units=2, activation=None)
dists = []
for i in range(0,num_actions):
    dists.append(tf.layers.dense(inputs=layer2, units=num_atoms, activation=tf.nn.softmax))

m_inp = tf.placeholder('float', [num_actions,None,num_atoms])
loss = -tf.reduce_sum( m_inp * tf.log(dists) )
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=m_inp,logits=dists)
optim = tf.train.AdamOptimizer(0.001).minimize(loss)

v_min = 0.0
v_max = 100.0
z_delta = (v_max-v_min)/float(num_atoms-1)
z_hist = [v_min + z_delta*float(i) for i in range(0,num_atoms)]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

memory = Memory(100000)


def calc_mprobs(ps,a,done,reward):
    # ps = np.vstack(ps)
    m_probs = np.zeros(num_atoms)
    if done:
        tz = min(v_max, max(v_min, reward ))
        b = (tz - v_min) / z_delta
        l = int(np.floor(b))
        u = int(np.ceil(b))
        m_probs[l] += (u-b)
        m_probs[u] += (b-l)
        return m_probs
    for j in range(0, num_atoms):
        t = reward + gamma*z_hist[j] 
        tz = min(v_max, max(v_min, t ))
        b = (tz - v_min) / z_delta
        l = int(np.floor(b))
        u = int(np.ceil(b))
        m_probs[l] += ps[a][j]*(u-b)
        m_probs[u] += ps[a][j]*(b-l)
    return m_probs

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
        if epsilon > 0 and rr > 50: epsilon *= 0.9999999

        obs_new,reward,done,info =  env.step(a)
        total += reward
   
        ps = np.vstack(sess.run( dists, feed_dict={state_inp: [obs_new]} ))
        memory.append( (obs,a,reward,obs_new,done,ps) )   

        targets=[]
        states=[]
        next_states=[]
        samples = memory.sample(None,batch_size) 
        targets=[]

        next_states = [s[3] for s in samples]
        pss = sess.run(dists, feed_dict={state_inp:next_states})
        s__ = np.transpose(pss, [1,0,2])

        m_prob = [np.zeros((batch_size, num_atoms)) for _ in range(num_actions)]
        k=0
        for (ps, (s_,a_,r_,sp_,done_,_)) in zip(s__,samples):
            q_expect = np.sum(np.multiply(np.vstack(ps), np.array(z_hist)), axis=1)
            opt_a = int(np.argmax( q_expect ))
            a_ = int(a_)
            if done_: 
                Tz = min(v_max, max(v_min, r_))
                bj = (Tz - v_min) / z_delta 
                m_l, m_u = np.floor(bj), np.ceil(bj)
                m_prob[a_][k][int(m_l)] += (m_u - bj)
                m_prob[a_][k][int(m_u)] += (bj - m_l)
            else:
                for j in range(num_atoms):
                    Tz = min(v_max, max(v_min, r_ + gamma * z_hist[j]))
                    bj = (Tz - v_min) / z_delta 
                    m_l, m_u = np.floor(bj), np.ceil(bj)
                    m_prob[a_][k][int(m_l)] += ps[opt_a][j] * (m_u - bj)
                    m_prob[a_][k][int(m_u)] += ps[opt_a][j] * (bj - m_l)
            k += 1

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
