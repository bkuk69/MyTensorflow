import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
sess = tf.Session()


W = tf.Variable(tf.random_uniform([1], -1.0,1.0 ))
b = tf.Variable(tf.random_uniform([1], -1.0,1.0 ))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

model = W * x + b

cost = tf.reduce_mean(tf.square(model - y))
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.005)

train = opt.minimize(cost)

train_tot = 10000
sess.run(tf.global_variables_initializer())
x_tr = [ 10,15,16, 1, 4, 6, 18, 12,14,7] #input
y_tr = [ 5, 2, 1,  9, 7, 8, 1,  5, 3, 6] #output

for i in range(train_tot) :
    error, _ = sess.run([cost, train], feed_dict={x: x_tr, y:y_tr})
    print(i, 'error = %.3f' %error, 'W= %.3f' %sess.run(W), 'b= %.3f' %sess.run(b)) # 애러가 줄어드는 방향으로 W는 1로 가까이가고 b는 0으로 가까이 갑니다.
test = 30
guess = sess.run(model, feed_dict={x:test})
print("\ntest=", test, "guess=%.3f" %guess)