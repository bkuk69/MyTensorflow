import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
sess = tf.Session()
j =[0.03, 0.03, 0.01, 0.9, 0.01, 0.01, 0.0025, 0.0025,
    0.0025, 0.0025]
k = [0,0,0,1,0,0,0,0,0,0]

log = -(tf.log(j))
print(sess.run(log))
prod = k * log
print(sess.run(prod))
i = tf.reduce_sum(prod)
print("cross entropy값",sess.run(i))