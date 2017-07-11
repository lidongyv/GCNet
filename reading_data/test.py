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

glfilenames=gfile.Glob(os.path.join(r'D:\stereo dataset\Stereo Matching\testpfm','*.tfrecords'))
gldisparity=tf.train.string_input_producer(glfilenames,shuffle=False)
reader=tf.TFRecordReader()
key,value=reader.read(gldisparity)
features=tf.parse_single_example(
        value,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label_raw': tf.FixedLenFeature([], tf.string),
          'name_raw': tf.FixedLenFeature([], tf.string),                                
      })

image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([540*960*3])
image=tf.reshape(image,[540,960,3])
label=tf.decode_raw(features['label_raw'], tf.float32)
label.set_shape([540*960])
label=tf.reshape(label,[540,960])
name=features['name_raw']
data=[image,label,name]
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
    a,b,c=[],[],[]
    d=data[0].eval()
    for i in range(18):
        #a.append(data[1].eval())
        b.append(data[1].eval())
        #c.append(name.eval().decode('UTF-8'))

    # Retrieve a single instance:
    #image= images.eval()
    win = cv2.namedWindow('test win'+str(i), flags=0)
    cv2.imshow('test win'+str(i),d.astype(np.uint8))
    cv2.waitKey()  
    coord.request_stop()
    coord.join(threads)
 