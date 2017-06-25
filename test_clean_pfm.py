# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:15:18 2017

@author: lidong
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
import cv2

glfilenames=gfile.Glob(os.path.join(r'D:\GC-Base','*.csv'))
gldisparity=tf.train.string_input_producer(glfilenames,shuffle=False)
reader=tf.WholeFileReader()
key,value=reader.read(gldisparity)
disparity=tf.decode_csv(value,record_defaults=tf.float32)

#images=tf.image.decode_png(value)
#images.set_shape([540,960,3])
#input_batch,labels=tf.train.shuffle_batch([[images,images],[key,key]],batch_size=3,capacity=2,min_after_dequeue=1)
epochs=1
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    #for i in range(3) :
  # Start populating the filename queue.
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    a=disparity.eval()
    for i in range(3) :
        print(key.eval())
    # Retrieve a single instance:
    #image= images.eval()
    #win = cv2.namedWindow('test win'+str(i), flags=0)
    #cv2.imshow('test win'+str(i), image.astype(np.uint8))
         
    coord.request_stop()
    coord.join(threads)
    cv2.waitKey(0)  