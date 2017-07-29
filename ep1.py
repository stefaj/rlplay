import gym
import numpy as np
import random
import tensorflow as tf

env = gym.make('FrozenLake-v0')

lr = .7
y = .95
num_episodes = 40000

def train_t(Q):
    #Initialize table with all zeros
    # Set learning parameters
    #create lists to contain total rewards and steps per episode
    #jList = []
    epsilon = 1.0
    rList = []
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            #Choose an action by greedily (with noise) picking from Q table
            if (random.random() < epsilon):
                # a = np.argmax(np.random.randn(1,env.action_space.n))
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s,:])
            if epsilon > 0.1:
                epsilon *= 0.99999
            print('eps', epsilon)
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            if d == True:
                break
        #jList.append(j)
        rList.append(rAll)
    print("Train: Score over time: " +  str(sum(rList)/num_episodes))
    print("Final Q-Table Values")
    print(Q)
    return Q

def play_t(Q):
    rList = []
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            a = np.argmax(Q[s,:]) 
            s1,r,d,_ = env.step(a)
            rAll += r
            s = s1
            if d == True:
                break
        #jList.append(j)
        rList.append(rAll)
    print("Play: Score over time: " +  str(sum(rList)/num_episodes))

# Q = np.zeros([env.observation_space.n,env.action_space.n])
# Q = train_t(Q)
# play_t(Q)



# neural net bois
tf.reset_default_graph()
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)



def train_q():
    epsilon = 1.0
    jList = []
    rList = []
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            while j < 99:
                j+=1

                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
                if (random.random() < epsilon):
                    a[0] = env.action_space.sample()

                if epsilon > 0.1:
                    epsilon *= 0.99999

                print('eps', epsilon)
                s1,r,d,_ = env.step(a[0])
                # Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})

                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + y*maxQ1

                _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})

                rAll += r
                s = s1

                if d == True:
                  #Reduce chance of random action as we train the model.
                  e = 1./((i/50) + 10)
                  break

            jList.append(j)
            rList.append(rAll)

    print("Train: Score over time: " +  str(sum(rList)/num_episodes))
    return init


def play_q(vs):
    epsilon = 1.0
    jList = []
    rList = []

    with tf.Session() as sess:
        sess.run(vs)
        for i in range(num_episodes):
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            while j < 99:
                j+=1

                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})

                s1,r,d,_ = env.step(a[0])

                # _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})

                rAll += r
                s = s1

                if d==True: break

            jList.append(j)
            rList.append(rAll)
    print("Train: Score over time: " +  str(sum(rList)/num_episodes))

vs = train_q()
play_q(vs)
