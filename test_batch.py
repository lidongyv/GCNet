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
from input_fn import *
IMG_WIDTH = 512 
IMG_HEIGHT = 256
IMG_CHAN=1
#load a single converted tfrecords
file=gfile.Glob(os.path.join(r'D:\SceneFlow','scene_flow_data_2.tfrecords'))
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
name=features['name_raw']
#reshape the data
limage.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT*COLOR_CHAN)
limage=tf.reshape(limage,[ORIGINAL_HEIGHT,ORIGINAL_WIDTH,COLOR_CHAN])
limage=tf.to_float(limage)/255
rimage.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT*COLOR_CHAN)
rimage=tf.reshape(rimage,[ORIGINAL_HEIGHT,ORIGINAL_WIDTH,COLOR_CHAN])
rimage=tf.to_float(rimage)/255

ldisparity.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT)
ldisparity=tf.reshape(ldisparity,[ORIGINAL_HEIGHT,ORIGINAL_WIDTH,IMG_CHAN])
rdisparity.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT)
rdisparity=tf.reshape(rdisparity,[ORIGINAL_HEIGHT,ORIGINAL_WIDTH,IMG_CHAN])
left=tf.concat([limage,ldisparity],axis=2)
right=tf.concat([rimage,rdisparity],axis=2)
left=tf.random_crop(left,[IMG_HEIGHT,IMG_WIDTH,4])
right=tf.random_crop(right,[IMG_HEIGHT,IMG_WIDTH,4])
[images,disparities]=tf.train.shuffle_batch([[left[:,:,0:3],right[:,:,0:3]],[left[:,:,3],right[:,:,3]]],batch_size=1,capacity=2,num_threads=4,min_after_dequeue=1)
images_s=tf.split(images,num_or_size_splits=2,axis=1)
limg_s=tf.reshape(images_s[0],[IMG_HEIGHT,IMG_WIDTH,3])
rimg_s=tf.reshape(images_s[1],[IMG_HEIGHT,IMG_WIDTH,3])
ground=tf.split(disparities,num_or_size_splits=2,axis=1)
lground=tf.reshape(ground[0],[IMG_HEIGHT,IMG_WIDTH])
rground=tf.reshape(ground[1],[IMG_HEIGHT,IMG_WIDTH])
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    #for i in range(3) :
  # Start populating the filename queue.
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image= (limg_s.eval()*255).astype(np.uint8)
    win = cv2.namedWindow('test win', flags=0)
    cv2.imshow('test win', image)
         
    coord.request_stop()
    coord.join(threads)
    cv2.waitKey(0)  