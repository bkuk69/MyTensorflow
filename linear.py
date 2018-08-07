import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
sess = tf.Session()


W = tf.Variable(tf.random_uniform([1], -1.0,1.0 ))
b = tf.Variable(tf.random_uniform([1], -1.0,1.0 ))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

model = W * x + b

cost = tf.reduce_mean(tf.square(model-y))
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.05)

x = tf.constant([[1., 1.],[2., 2.]])
tf.reduce_mean(x,0)
tf.reduce_mean(x,1)
tf.reduce_mean(x)

train = opt.minimize(cost)

train_tot = 100
sess.run(tf.global_variables_initializer())

x_tr = [1,2,3]
y_tr = [1,2,3]

for i in range(train_tot) :
    error, _ = sess.run([cost, train], feed_dict={x: x_tr, y:y_tr})
    print(i, 'error = %.3f' %error, 'W= %.3f' %sess.run(W), 'b= %.3f' %sess.run(b))

test =5
guess = sess.run(model, feed_dict={x,test})
print("\ntest=", test, "guess = %.3f" % guess)