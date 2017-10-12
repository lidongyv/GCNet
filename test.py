# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:33:47 2017

@author: lidong
"""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

ORIGINAL_WIDTH = 960
ORIGINAL_HEIGHT = 540
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 960 
IMG_HEIGHT = 540
IMG_CHAN=1

file=gfile.Glob(os.path.join(r'D:\KITTI2015','KITTI2015_train.tfrecords'))

data=tf.train.string_input_producer(file,shuffle=False)
reader=tf.TFRecordReader()
key,value=reader.read(data)
features=tf.parse_single_example(
    value,
    features={
    'name_raw':        tf.FixedLenFeature([], tf.string),                                
  })


name=features['name_raw']
#reshape the data

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    #for i in range(3) :
  # Start populating the filename queue.
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    a=name.eval()
    print(a.decode('UTF-8'))    
    # Retrieve a single instance:
    #image= images.eval()
    #win = cv2.namedWindow('test win'+str(i), flags=0)
    #cv2.imshow('test win'+str(i), image.astype(np.uint8))
    #cv2.waitKey(0)    
    coord.request_stop()
    coord.join(threads)
b=a.decode('UTF-32')
print(b)
