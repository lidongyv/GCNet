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

from tensorflow.python.training import moving_averages


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
		self.mode='train'
		self._extra_train_ops = []
		self._extra_train_ops=[]

	def build_graph(self):
		self.global_setp=tf.contrib.framework.get_or_create_global_step()
		self._build_model()
		self._build_train_op()
		self.summaries=tf.summary.merge_all()
	def _sia_conv(self,input,kernel_size,stride):
		#convolution for siamese
		n=kernel[0]*kernel[1]*kernel[3]
		weights=tf.get_variable('weights',kernel_size, tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
		#biases=tf.get_variable('biases',bias_shape,initializer=tf.constant_initializer(0.0))
		conv=tf.nn.conv2d(input,weights,strides=stride,padding='SAME')
		return conv	

	def _conv3d(self, input,kernel,stride=[1,1,1,1,1]):
		n=kernel[0]*kernel[1]*kernel[2]*kernel[4]
		weights=tf.get_variable('weights',kernel,tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
		conv=tf.nn.conv3d(input,weights,strides=stride,padding='SAME')
		conv=self._batch_norm(conv)
		conv=self._relu(conv)
		return conv

	def _deconv3d(self, input,kernel,stride=[1,1,1,1,1]):
		n=kernel[0]*kernel[1]*kernel[2]*kernel[3]
		weights=tf.get_variable('weights',kernel,tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
		conv=tf.nn.deconv3d(input,weights,strides=stride,padding='SAME')
		conv=self._batch_norm(conv)
		conv=self._relu(conv)
		return conv
	def _relu(self, x, leakiness=0.0):
		"""Relu, with optional leaky support."""
		return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


	def _residual_cnn(self,x,in_filter,out_filter,stride,activate_before_residual=True):
		if activate_before_residual:
			with tf.variable_scope('shared_activation'):
				x = self._batch_norm('init_bn', x)
				x = self._relu(x, 0)
				orig_x = x
		else:
			with tf.variable_scope('residual_only_activation'):
				orig_x = x
				x = self._batch_norm('init_bn', x)
				x = self._relu(x, 0)

		with tf.variable_scope('res1'):
			kernel=[3,3,32,32]
			bias=[32]
			stride=[1,1,1,1]
			out=self._sia_conv(out,kernel,stride)
		with tf.variable_scope('res2'):
			out=self._batch_norm('bn2',out)
			out=self._relu(out,0)
			kernel=[3,3,32,32]
			bias=[32]
			stride=[1,1,1,1]
			out=self._sia_conv(out,kernel,stride)
		with tf.variable_scope('res_add'):
			out=orig_x+out
		return out


	def _batch_norm(self, name, x):
		"""Batch normalization."""
		with tf.variable_scope(name):
			params_shape = [x.get_shape()[-1]]

			beta = tf.get_variable(
	         	 'beta', params_shape, tf.float32,
	         	 initializer=tf.constant_initializer(0.0, tf.float32))
			gamma = tf.get_variable(
	         	 'gamma', params_shape, tf.float32,
	       	   initializer=tf.constant_initializer(1.0, tf.float32))

			if self.mode == 'train':
				mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

				moving_mean = tf.get_variable(
		            'moving_mean', params_shape, tf.float32,
		            initializer=tf.constant_initializer(0.0, tf.float32),
		            trainable=False)
				moving_variance = tf.get_variable(
		            'moving_variance', params_shape, tf.float32,
		            initializer=tf.constant_initializer(1.0, tf.float32),
		            trainable=False)

				self._extra_train_ops.append(moving_averages.assign_moving_average(
		            moving_mean, mean, 0.9))
				self._extra_train_ops.append(moving_averages.assign_moving_average(
		            moving_variance, variance, 0.9))
			else:
				mean = tf.get_variable(
		            'moving_mean', params_shape, tf.float32,
		            initializer=tf.constant_initializer(0.0, tf.float32),
		            trainable=False)
				variance = tf.get_variable(
		            'moving_variance', params_shape, tf.float32,
		            initializer=tf.constant_initializer(1.0, tf.float32),
		            trainable=False)
				tf.summary.histogram(mean.op.name, mean)
				tf.summary.histogram(variance.op.name, variance)
		      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
			y = tf.nn.batch_normalization(
		          x, mean, variance, beta, gamma, 0.001)
			y.set_shape(x.get_shape())
		return y
	def siamese_cnn(self,image):
		with device('/gpu:0'):
			with tf.variable_scope('conv1'):
				kernel=[5,5,3,32]
				bias=[32]
				stride=[1,2,2,1]
				out=self._sia_conv(image,kernel,stride,bias)
			#resnet 2d conv part
			for i in range(8):
				with tf.variable_scope('res_uint_%d' % i):
					out=self._residual_cnn(out)
			out=self._batch_norm('conv17_bn',out)
			out=self._relu(out,0)
			with tf.variable_scope('conv18'):
				kernel=[3,3,32,32]
				bias=[32]
				stride=[1,1,1,1]
				out=self._sia_conv(out,kernel,stride)
	def _3d_cnn(self,input):
		with device('/gpu:0'):
			with tf.variable_scope('conv19'):
				kernel=[3,3,3,64,32]
				stride=[1,1,1,1,1]
				out=self._conv3d(input,kernel,stride)
			with tf.variable_scope('conv20'):
				kernel=[3,3,3,32,32]
				stride=[1,1,1,1,1]
				out20=self._conv3d(out,kernel,stride)
			with tf.variable_scope('conv21'):
				kernel=[3,3,3,32,64]
				stride=[1,2,2,2,1]
				out=self._conv3d(out20,kernel,stride)
			with tf.variable_scope('conv22'):
				kernel=[3,3,3,64,64]
				stride=[1,1,1,1,1]
				out=self._conv3d(out,kernel,stride)
			with tf.variable_scope('conv23'):
				kernel=[3,3,3,64,64]
				stride=[1,1,1,1,1]
				out23=self._conv3d(out,kernel,stride)
			with tf.variable_scope('conv24'):
				kernel=[3,3,3,64,64]
				stride=[1,2,2,2,1]
				out=self._conv3d(out23,kernel,stride)
			with tf.variable_scope('conv25'):
				kernel=[3,3,3,64,64]
				stride=[1,1,1,1,1]
				out=self._conv3d(out,kernel,stride)
			with tf.variable_scope('conv26'):
				kernel=[3,3,3,64,64]
				stride=[1,1,1,1,1]
				out26=self._conv3d(out,kernel,stride)
			with tf.variable_scope('conv27'):
				kernel=[3,3,3,64,64]
				stride=[1,2,2,2,1]
				out=self._conv3d(out26,kernel,stride)
			with tf.variable_scope('conv28'):
				kernel=[3,3,3,64,64]
				stride=[1,1,1,1,1]
				out=self._conv3d(out27,kernel,stride)
			with tf.variable_scope('conv29'):
				kernel=[3,3,3,64,64]
				stride=[1,1,1,1,1]
				out29=self._conv3d(out,kernel,stride)
			with tf.variable_scope('conv30'):
				kernel=[3,3,3,64,128]
				stride=[1,2,2,2,1]
				out=self._conv3d(out29,kernel,stride)
			with tf.variable_scope('conv31'):
				kernel=[3,3,3,128,128]
				stride=[1,1,1,1,1]
				out=self._conv3d(out,kernel,stride)
			with tf.variable_scope('conv32'):
				kernel=[3,3,3,128,128]
				stride=[1,1,1,1,1]
				out=self._conv3d(out,kernel,stride)
			with tf.variable_scope('de_conv33'):
				kernel=[3,3,3,64,128]
				stride=[1,2,2,2,1]
				out=self._deconv3d(out,kernel,stride)
				out+=out29
			with tf.variable_scope('de_conv34'):
				kernel=[3,3,3,64,64]
				stride=[1,2,2,2,1]
				out=self._deconv3d(out,kernel,stride)
				out+=out26
			with tf.variable_scope('de_conv35'):
				kernel=[3,3,3,64,64]
				stride=[1,2,2,2,1]
				out=self._deconv3d(out,kernel,stride)
				out+=out23
			with tf.variable_scope('de_conv36'):
				kernel=[3,3,3,32,64]
				stride=[1,2,2,2,1]
				out=self._deconv3d(out,kernel,stride)
				out+=out20
			with tf.variable_scope('de_conv37'):
				kernel=[3,3,3,1,32]
				stride=[1,2,2,2,1]
				n=kernel[0]*kernel[1]*kernel[2]*kernel[3]
				weights=tf.get_variable('weights',kernel,tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
				out=tf.nn.deconv3d(out,weights,strides=stride,padding='SAME')
			return out


	def _softargmin(disparity):
		disparity=-disparity
		probability=tf.nn.softmax(disparity)
		sum=0
		for d in range(192):
			sum+=d*probability[d]
		return sum
	def _todisparity(volume):
		with tf.variable_scope('soft_argmin'):
			disparity=tf.get_variable('disparity',[540,960],tf.float32,initializer=tf.constant_initializer())
			for i in range(540):
				for j in range(960):
					disparity[i][j]=self._softargmin(volume[:,i,j])

	def create_volume(linput,rinput):
		#create the cost volume from w*h*f to d*w*h*2*f
		#using split and concat to achieve the translation
		lvolume=[]
		for i in range(96):
			splits=tf.split(rinput,[96-i,i],axis=1)
			rsplits=tf.concat(split[1],split[0],axis=1)
			single_dis=tf.concat([linput,rsplits],3)
			volume.append(single_dis)
		lvol=tf.stack(volume,axis=0)
		rvolume=[]
		for i in range(96):
			splits=tf.split(linput,[i,96-i],axis=1)
			lsplits=tf.concat(split[1],split[0],axis=1)
			single_dis=tf.concat([rinput,lsplits],3)
			volume.append(single_dis)
		rvol=tf.stack(rvolume,axis=0)
		return lvol,rvol

	def _loss(lpre,rpre):
		ground=tf.split(self.labels,num_or_size_splits=2,axis=1)
		lground=tf.reshape(ground[0],[1,540,960])
		rground=tf.reshape(ground[1],[1,540,960])
		sum=0
		for i in range(1):
			for x in range(540):
				for y in range(960):
					sum+=tf.abs(lpre[i,x,y]-lground[i,x,y])
		loss=sum/(540*960)
		sum=0
		for i in range(1):
			for x in range(540):
				for y in range(960):
					sum+=tf.abs(rpre[i,x,y]-rground[i,x,y])
		loss=loss+sum/(960*540)
		return loss
	def _build_model(self):
		images=tf.split(self.image,num_or_size_splits=2, axis=1)
		limage=tf.reshape(images[0],[1,540,960,3])
		rimage=tf.reshape(image[1],[1,540,960,3])
		#residual siamese convolution
		with tf.variable_scope('siamese_conv') as scope:
			left=self.siamese_cnn(limage)
			scope.reuse_variables()
			right=self.siamese_cnn(rimage)
		#create the cost volume from w*h*f to d*w*h*(2*f)
		left,right=self.create_volume(left,right)
		left=tf.reshape(left,[192,540,960])
		right=tf.reshape(right,[192,540,960])
		#3d convolution
		with tf.variable_scope('3d_conv') as scope:
			left=self._3d_cnn(left)
			scope.reuse_variables()
			right=self._3d_cnn(right)
		with tf.variable_scope('soft_argmin'):
			ldisparities=self._todisparity(left)
			scope.reuse_variables()
			rdisparities=self._todisparity(right)
		self.lpre=ldisparities
		self.rpre=rdisparities
		self.loss=self._loss(ldisparities,rdisparities)
	def _build_train_op(self):
		"""Build training specific ops for the graph."""
		self.lrn_rate = tf.constant(0.0001, tf.float32)
		tf.summary.scalar('learning_rate', self.lrn_rate)

		trainable_variables = tf.trainable_variables()
		grads = tf.gradients(self.cost, trainable_variables)
		optimizer = tf.train.RMSPropOptimizer(
			self.lrn_rate,
			decay=0.9,
			momentum=0.9,
			epsilon=1.0)


		apply_op = optimizer.apply_gradients(
			zip(grads, trainable_variables),
			global_step=self.global_step, name='train_step')

		train_ops = [apply_op] + self._extra_train_ops
		self.train_op = tf.group(*train_ops)





		















