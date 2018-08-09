import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sess = tf.Session()

char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']

num_dic ={n: i for i, n in enumerate(char_arr)}


#print(num_dic)

seq_data =['body','dial', 'open', 'rank', 'need', 'wise', 'item', 'jury', 'path', 'ease']

n_input = n_class = 26
n_stage = 3

def make_batch(seq_data):
    input_batch =[]
    target_batch =[]

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0:-1] ]
        target = num_dic[seq[3]]
        input_batch.append(np.eye(26)[input])
        target_batch.append(target)
    return input_batch, target_batch
#x, y = make_batch(seq_data)
#print(x)
#print(y)