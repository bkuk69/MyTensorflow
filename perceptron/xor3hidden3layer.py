import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
sess = tf.Session()

T, F = 1., -1.
bias = 1.

train_in= [
    [T, T, T],
    [T, T, F],
    [T, F, T],
    [T, F, F],
    [F, T, T],
    [F, T, F],
    [F, F, T],
    [F, F, F]
]

train_out = [[T], [F], [F], [T], [F], [T], [T], [F]]
# B4: hidden layer 1 definition
w1 = tf.Variable(tf.random_normal([3, 3]))
b1 = tf.Variable(tf.zeros([3]))
out1 = tf.tanh(tf.add(tf.matmul(train_in, w1), b1))

# B4: hidden layer 2 definition
w2 = tf.Variable(tf.random_normal([3, 3]))
b2 = tf.Variable(tf.zeros([3]))
out2 = tf.tanh(tf.add(tf.matmul(out1, w2), b2))

# B4: hidden layer 2 definition
w3 = tf.Variable(tf.random_normal([3, 3]))
b3 = tf.Variable(tf.zeros([3]))
out3 = tf.tanh(tf.add(tf.matmul(out2, w3), b3))

# B5: output layer definition
w4 = tf.Variable(tf.random_normal([3, 1]))
b4 = tf.Variable(tf.zeros([1]))
out4 = tf.tanh(tf.add(tf.matmul(out3, w4), b4))

# B6: error calculation
error = tf.subtract(train_out, out4)
mse = tf.reduce_mean(tf.square(error))

# B7: training objective
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

# B8: tensorflow session preparation
sess = tf.Session()
sess.run(tf.global_variables_initializer())
err, target = 1, 0.01
epoch, max_epochs = 0, 1000

# B9: print W, b, and sample result
def test():
    print('\nweight 1\n', sess.run(w1))
    print('bias 1\n', sess.run(b1))
    print('\nweight 2\n', sess.run(w2))
    print('bias 2\n', sess.run(b2))
    print('weight 3\n', sess.run(w3))
    print('bias 3\n', sess.run(b3))
    print('\nweight 4\n', sess.run(w4))
    print('bias 4\n', sess.run(b4))
    print('output\n', sess.run(out4))
    print('mse: ', sess.run(mse), '\n')

# B10: main session
test()
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, ', mse:', err)
test()
