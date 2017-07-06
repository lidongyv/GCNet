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
import cv2

import python_pfm
#FLAGS=flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 540
ORIGINAL_HEIGHT = 960
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 64
IMG_HEIGHT = 64

def get_input(mode=0):
    """creat input data and ground truth data for network
    Args:
    	the mode is training or prediction
    Return:
    	three matrix for left images, right images, with the conresponding ground truth images
    """
    #load a single converted tfrecords
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
    limage=tf.to_float(limage)/255
    rimage.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT*COLOR_CHAN)
    rimage=tf.reshape(rimage,[ORIGINAL_WIDTH,ORIGINAL_HEIGHT,COLOR_CHAN])
    rimage=tf.to_float(rimage)/255
    ldisparity.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT)
    ldisparity=tf.reshape(ldisparity,[ORIGINAL_WIDTH,ORIGINAL_HEIGHT])

    rdisparity.set_shape(ORIGINAL_WIDTH*ORIGINAL_HEIGHT)
    rdisparity=tf.reshape(rdisparity,[ORIGINAL_WIDTH,ORIGINAL_HEIGHT])

    [input_batch,disaprity_batch]=tf.train.shuffle_batch([[limage,rimage],[ldisparity,rdisparity]],batch_size=1,capacity=5,min_after_dequeue=1)
    return input_batch,disaprity_batch
    
    






    
