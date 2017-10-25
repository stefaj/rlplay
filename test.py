import tensorflow as tf
import numpy as np

W = tf.Variable([3], dtype=tf.float32)
b = tf.Variable([-2.5], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

## fixW = tf.assign(W,[1.9])
#fixb = tf.assign(b,[1])
#sess.run([fixW, fixb])

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(0,1000):
    xs = [2.0,4.0,5.0]
    ys = [x_*2.0+1.0 for x_ in xs]
    sess.run(train, {x:xs, y:ys})

ans = sess.run([W,b,loss], {x:[1,2,3,4], y:[3,5,7,9]})

print(ans)
