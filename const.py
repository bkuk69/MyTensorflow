import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
sess = tf.Session()
a = tf.constant(2)
b = tf.constant(3)
x= tf.add(a,b)
print(sess.run(x))