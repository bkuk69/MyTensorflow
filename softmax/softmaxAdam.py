import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST", one_hot="True")
sess = tf.Session()

x = tf.placeholder(tf.float32, [None, 784]) #input neuron은 28*28이다.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b ) #추측값

ans = tf.placeholder(tf.float32, [None, 10]) #정답
loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(y),1))
opt = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

sess.run(tf.global_variables_initializer())

def test():
    x_train = mnist.test.images[0:1, 0:784]
    answer = sess.run(y, feed_dict={x:x_train})
    print("\ny vector is ", answer)
    print("my guess is ", answer.argmax())#제일큰 숫자가 몇번째 있는지 나타냄

train_tot = 10000
batch_size = 100

test()
for i in range(train_tot):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    error, _ = sess.run([loss, opt], feed_dict={x:batch_xs, ans:batch_ys})
    if(i % 100 == 0):
        print("batch ", i, "error = %.3f" % error)
test()

correct = tf.equal(tf.argmax(y, 1), tf.argmax(ans, 1)) #1번째 축으로 해라. 즉 y축으로 해야 각 사진의 추측치 중 가장 큰값을 가져다 준다.
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
images = mnist.test.images
labels = mnist.test.labels
print("\nmodel accuracy: ", sess.run(accuracy, feed_dict={x: images, ans:labels}))