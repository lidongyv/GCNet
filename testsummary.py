# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:49:56 2017

@author: lidong
"""

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
import model as whole_model
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

def train():
	tf.device('/cpu:0')
	#get input data
	lpre=tf.constant(1.0)
	rpre=tf.constant(2.0)
	global_step = tf.contrib.framework.get_or_create_global_step()
	summary_hook = tf.train.SummarySaverHook(
      save_steps=10,
      output_dir=r'D:\GC-Base\log\output',
      summary_op=tf.summary.merge([tf.summary.scalar('lpre',lpre),tf.summary.scalar('rpre',rpre)]))
	logging_hook = tf.train.LoggingTensorHook(
      tensors={'lpre': lpre,
               'rpre': rpre,
               'global_step':global_step,

               },
      every_n_iter=10)
	b=lpre*rpre
	with tf.train.MonitoredTrainingSession(
      checkpoint_dir=r'D:\GC-Base\log',
      hooks=[logging_hook],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=None,
      save_checkpoint_secs=None,
      config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as mon_sess:
		while not mon_sess.should_stop():
			mon_sess.run(b)
			steps=global_step.eval(session=mon_sess)
			print('running'+str(steps))
			"""
			if setps>1 and model.save==1:
				b_summary_op=tf.summary.merge([model.summaries,
      							tf.summary.image('lpre',model.lpre,max_outputs=1),tf.summary.image('rpre',model.lpre,max_outputs=1)])
				saver = tf.train.Saver(b_summary_op)	
				saver.save(mon_sess,'best_model',global_step=steps)
			"""

train()

	 
 
 
 
 
 
 
 