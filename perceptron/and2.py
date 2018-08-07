import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
sess = tf.Session()

T, F = 1, -1
bias = 1.

train_in= [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias]
]

train_out = [[T], [F], [F], [F]]