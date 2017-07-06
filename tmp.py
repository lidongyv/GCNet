# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:17:51 2017

@author: lidong
"""
import tensorflow as tf
# Creates a graph.
def d2(input):
    return 1
c = []
for d in ['/gpu:2', '/gpu:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
init_op = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init_op)
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i,j].eval(session=sess))


    