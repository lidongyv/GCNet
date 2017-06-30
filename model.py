# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:00:20 2017

@author: lidong
"""

import tensorflow as tf 
import cv2
import numpy as np 
import argparse
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

ORIGINAL_WIDTH = 540
ORIGINAL_HEIGHT = 960
COLOR_CHAN = 3

class E2EModel(object):
	def __init__(self,
				 image=None,
				 groundtruth=None,
				 reuse_scope=None
		):

		self.image=image
		self.labels=groundtruth

	def build_graph(self):
		self.global_setp=tf.contrib.framework.get_or_create_global_step()
		self._build_model()
		self._build_train_op()
		self.summaries=tf.summary.merge_all()

	def _build_model(self):
		images=tf.split(self.image,num_or_size_splits=2, axis=1)
		limage=tf.reshape(images[0],[6,540,960,3])
		rimage=tf.reshape(image[1],[6,540,960,3])
		disparities=tf.split(self.labels,num_or_size_splits=2,axis=1)
		ldisparities=tf.reshape(disparities[0],[6,540,960])
		rdisparities=tf.reshape(disparities[1],[6,540,960])
















