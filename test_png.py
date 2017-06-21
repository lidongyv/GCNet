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


olfilenames=gfile.Glob(os.path.join(r'D:\stereo dataset\Stereo Matching\frames_cleanpass','*','left','*.png'))
olimages=tf.train.string_input_producer([olfilenames[1]],shuffle=False)
reader=tf.WholeFileReader()
key,value=reader.read(olimages)
images=tf.image.decode_png(value)
epochs=1
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # Start populating the filename queue.
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1):
    # Retrieve a single instance:
      image = images.eval()
    print(image.shape) 
    
    coord.request_stop()
    coord.join(threads)