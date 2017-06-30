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

ORIGINAL_WIDTH = 540
ORIGINAL_HEIGHT = 960
COLOR_CHAN = 3

file=gfile.Glob(os.path.join(r'D:\SceneFlow','tmp_data.tfrecords'))
data=tf.train.string_input_producer(file,shuffle=False)
reader=tf.TFRecordReader()
key,value=reader.read(data)
features=tf.parse_single_example(
    value,
    features={
    'image_left_raw':  tf.FixedLenFeature([], tf.string),
    'image_right_raw': tf.FixedLenFeature([], tf.string),
    'label_left_raw':  tf.FixedLenFeature([], tf.string),
    'label_right_raw': tf.FixedLenFeature([], tf.string),
    'name_raw':        tf.FixedLenFeature([], tf.string),                                
  })
#decode the data into image and disparity
limage=tf.decode_raw(features['image_left_raw'],tf.uint8)
rimage=tf.decode_raw(features['image_right_raw'],tf.uint8)
ldisparity=tf.decode_raw(features['label_left_raw'],tf.float32)
rdisparity=tf.decode_raw(features['label_right_raw'],tf.float32)
#reshape the data
limage.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT*COLOR_CHAN)
limage=tf.reshape(limage,[ORIGINAL_WIDTH,ORIGINAL_HEIGHT,COLOR_CHAN])

rimage.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT*COLOR_CHAN)
rimage=tf.reshape(rimage,[ORIGINAL_WIDTH,ORIGINAL_HEIGHT,COLOR_CHAN])

ldisparity.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT)
ldisparity=tf.reshape(ldisparity,[ORIGINAL_WIDTH,ORIGINAL_HEIGHT])

rdisparity.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT)
rdisparity=tf.reshape(rdisparity,[ORIGINAL_WIDTH,ORIGINAL_HEIGHT])
[input_batch,disaprity_batch]=tf.train.shuffle_batch([[limage,rimage],[ldisparity,rdisparity]],batch_size=6,capacity=5,min_after_dequeue=1)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    #for i in range(3) :
  # Start populating the filename queue.
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    a=input_batch.eval()
    # Retrieve a single instance:
    #image= images.eval()
    #win = cv2.namedWindow('test win'+str(i), flags=0)
    #cv2.imshow('test win'+str(i), image.astype(np.uint8))
    #cv2.waitKey(0)    
    coord.request_stop()
    coord.join(threads)
