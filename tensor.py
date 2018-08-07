import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
sess = tf.Session()
x = tf.placeholder(tf.float32,[None,3])
#W = tf.Variable([0.1,0.1],[0.1,0.1],[0.1,0.1])
#b = tf.Variable([0.2,0.2])
W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))
output = tf.matmul(x,W) + b
sess.run(tf.global_variables_initializer())
input = [[1,2,3],[4,5,6]]

print("input: ", input)
print("W: ", sess.run(W))
print("b: ", sess.run(b))
print("output: " , sess. run(output, feed_dict={x: input}))

print("shape of W:", W.get_shape())
print("shape of b:", b.get_shape())
print("shape of x:", x.get_shape())
print("shape of output:", output.get_shape())#input이 두개니까 2 나오는 것은 두개니까 2:2
#print("shape of input:", input.get_shape())#input은 리스트이지 tensorflow에 걸리지 않았으니까 get_shape()을 사용할 수 없음

import numpy as np
