# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 15:51:20 2017

@author: lidong
"""
import tensorflow as tf 
import cv2
import numpy as np 
import argparse
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from input_fn import *
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000 

# tf record data location:
DATA_DIR = 'push/push_train'

# local output directory
OUT_DIR = '/tmp/data'
"""
FLAGS = flags.FLAGS

flags.DEFINE_integer('mode',0,'0:prediction, 1:training with existing model, 2:training with new model')
flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '','filepath of a pretrained model to initialize from.')
flags.DEFINE_integer('sequence_length', 10,'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_string('model', '','model path for pretrained model')
flags.DEFINE_integer('batch_size', 1, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.001,'the base learning rate of the generator')
flags.DEFINE_string('vdata','', 'validation data')


def main(unused_args):
	images,disparities=get_input(FLAGS.mode)


if __name__ == '__main__':
  app.run()
 """
#get input data
images,disparities=get_input(1) 
#E2ENet=E2EModel(images,disparities)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image=images.eval()
    disparitiy=disparities.eval()
    coord.request_stop()
    coord.join(threads)
 
 
 
 
 
 
 
 