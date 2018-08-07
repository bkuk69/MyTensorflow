import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
sess = tf.Session()

T, F = 1., -1.
bias = 1.

train_in= [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias]
]

train_out = [[F], [T], [T], [F]]

w = tf.Variable(tf.random_normal([3,1]))

def step(x):
    is_greater = tf.greater(x,0) #return 타입이 bool형 x가 0보다 크냐 아니냐 알아봄 

    as_float = tf.to_float(is_greater) #실수형으로 만들어줌 파이썬은 True이면 1 False이면 0됨(C와 비슷))
    doubled = tf.multiply(as_float, 2) #2를 곱함
    return tf.subtract(doubled, 1)

output = step(tf.matmul(train_in, w))
error = tf.subtract(train_out, output) 
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(train_in, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess.run(tf.global_variables_initializer())
err, target = 1,0
epoch, max_epochs = 0, 1000

def test():
    print("\nweight/bias\n", sess.run(w))
    print("output\n", sess.run(output))
    print("mse: ", sess.run(mse),"\n")

test()
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print("epoch: ", epoch,"mse: ", err)
test()