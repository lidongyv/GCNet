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


IMG_WIDTH = 512
IMG_HEIGHT = 256
IMG_DISPARITY = 192

class E2EModel(object):
	def __init__(self,
				 image=None,
				 groundtruth=None,
				 mode=None
		):

		self.image=image
		self.labels=groundtruth
		self.mode=mode
		self._extra_train_ops = []
		self.lrn_rate=0.0001

	def build_graph(self):
		self.global_step=tf.contrib.framework.get_or_create_global_step()
		self._build_model()
		self._build_train_op()
		self.summaries=tf.summary.merge_all()
	def _sia_conv(self,input,kernel_size,stride):
		#convolution for siamese
		n=kernel_size[0]*kernel_size[1]*kernel_size[3]
		weights=tf.get_variable('weights',kernel_size, tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
		#biases=tf.get_variable('biases',bias_shape,initializer=tf.constant_initializer(0.0))
		conv=tf.nn.conv2d(input,weights,strides=stride,padding='SAME')
		return conv	

	def _conv3d(self,name,input,kernel,stride=[1,1,1,1,1]):
		n=kernel[0]*kernel[1]*kernel[2]*kernel[4]
		weights=tf.get_variable('weights',kernel,tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
		conv=tf.nn.conv3d(input,weights,strides=stride,padding='SAME')
		conv=self._batch_norm_3d(name+'_bn',conv)
		conv=self._relu(conv)
		return conv

	def _deconv3d(self,name, input,kernel,output,stride=[1,1,1,1,1]):
		n=kernel[0]*kernel[1]*kernel[2]*kernel[3]
		weights=tf.get_variable('weights',kernel,tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
		conv=tf.nn.conv3d_transpose(input,weights,output,strides=stride,padding='SAME')
		conv=self._batch_norm_3d(name,conv)
		conv=self._relu(conv)
		return conv

	def _relu(self, x, leakiness=0.0):
		"""Relu, with optional leaky support."""
		return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


	def _residual_cnn(self,x,activate_before_residual=True):
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
			out=self._sia_conv(x,kernel,stride)
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
	def _batch_norm_3d(self, name, x):
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
				mean, variance = tf.nn.moments(x, [0, 1, 2,3], name='moments')

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
		#with tf.device('/gpu:0'):
		with tf.variable_scope('conv1'):
			kernel=[5,5,3,32]
			bias=[32]
			stride=[1,2,2,1]
			out=self._sia_conv(image,kernel,stride)
		#resnet 2d conv part
		for i in range(8):
			with tf.variable_scope('res_uint_%d' % i):
				out=self._residual_cnn(out)
	#with tf.device('/gpu:0'):
		out=self._batch_norm('conv17_bn',out)
		out=self._relu(out,0)
		with tf.variable_scope('conv18'):
			kernel=[3,3,32,32]
			bias=[32]
			stride=[1,1,1,1]
			out=self._sia_conv(out,kernel,stride)
		return out
	def _3d_cnn(self,input):
		
		#with tf.device('/gpu:0'):
		with tf.variable_scope('conv19'):
			kernel=[3,3,3,64,32]
			stride=[1,1,1,1,1]
			out=self._conv3d('conv19',input,kernel,stride)
		with tf.variable_scope('conv20'):
			kernel=[3,3,3,32,32]
			stride=[1,1,1,1,1]
			out20=self._conv3d('conv20',out,kernel,stride)
		
		with tf.variable_scope('conv21'):
			kernel=[3,3,3,32,64]
			stride=[1,2,2,2,1]
			out=self._conv3d('conv21',out20,kernel,stride)
		with tf.variable_scope('conv22'):
			kernel=[3,3,3,64,64]
			stride=[1,1,1,1,1]
			out=self._conv3d('conv22',out,kernel,stride)
		with tf.variable_scope('conv23'):
			kernel=[3,3,3,64,64]
			stride=[1,1,1,1,1]
			out23=self._conv3d('conv23',out,kernel,stride)
		with tf.variable_scope('conv24'):
			kernel=[3,3,3,64,64]
			stride=[1,2,2,2,1]
			out=self._conv3d('conv24',out23,kernel,stride)
		with tf.variable_scope('conv25'):
			kernel=[3,3,3,64,64]
			stride=[1,1,1,1,1]
			out=self._conv3d('conv25',out,kernel,stride)
		with tf.variable_scope('conv26'):
			kernel=[3,3,3,64,64]
			stride=[1,1,1,1,1]
			out26=self._conv3d('conv26',out,kernel,stride)
		with tf.variable_scope('conv27'):
			kernel=[3,3,3,64,64]
			stride=[1,2,2,2,1]
			out=self._conv3d('conv27',out26,kernel,stride)

		with tf.variable_scope('conv28'):
			kernel=[3,3,3,64,64]
			stride=[1,1,1,1,1]
			out=self._conv3d('conv28',out,kernel,stride)
		with tf.variable_scope('conv29'):
			kernel=[3,3,3,64,64]
			stride=[1,1,1,1,1]
			out29=self._conv3d('conv29',out,kernel,stride)
		with tf.variable_scope('conv30'):
			kernel=[3,3,3,64,128]
			stride=[1,2,2,2,1]
			out=self._conv3d('conv30',out29,kernel,stride)
		with tf.variable_scope('conv31'):
			kernel=[3,3,3,128,128]
			stride=[1,1,1,1,1]
			out=self._conv3d('conv31',out,kernel,stride)
		with tf.variable_scope('conv32'):
			kernel=[3,3,3,128,128]
			stride=[1,1,1,1,1]
			out=self._conv3d('conv32',out,kernel,stride)
		with tf.variable_scope('de_conv33'):
			kernel=[3,3,3,64,128]
			stride=[1,2,2,2,1]
			output=tf.constant([1,np.ceil(IMG_DISPARITY/16).astype('int32'),np.ceil(IMG_HEIGHT/16).astype('int32'),np.ceil(IMG_WIDTH/16).astype('int32'),64])
			out=self._deconv3d('de_conv33',out,kernel,output,stride)
			out+=out29
		with tf.variable_scope('de_conv34'):
			kernel=[3,3,3,64,64]
			stride=[1,2,2,2,1]
			output=tf.constant([1,np.ceil(IMG_DISPARITY/8).astype('int32'),np.ceil(IMG_HEIGHT/8).astype('int32'),np.ceil(IMG_WIDTH/8).astype('int32'),64])
			out=self._deconv3d('de_conv34',out,kernel,output,stride)
			out+=out26
		with tf.variable_scope('de_conv35'):
			kernel=[3,3,3,64,64]
			stride=[1,2,2,2,1]
			output=tf.constant([1,np.ceil(IMG_DISPARITY/4).astype('int32'),np.ceil(IMG_HEIGHT/4).astype('int32'),np.ceil(IMG_WIDTH/4).astype('int32'),64])
			out=self._deconv3d('de_conv35',out,kernel,output,stride)
			out+=out23
		
		with tf.variable_scope('de_conv36'):
			kernel=[3,3,3,32,64]
			stride=[1,2,2,2,1]
			output=tf.constant([1,np.ceil(IMG_DISPARITY/2).astype('int32'),np.ceil(IMG_HEIGHT/2).astype('int32'),np.ceil(IMG_WIDTH/2).astype('int32'),32])
			out=self._deconv3d('de_conv36',out,kernel,output,stride)
			out+=out20
			
		with tf.variable_scope('de_conv37'):
			kernel=[3,3,3,1,32]
			stride=[1,2,2,2,1]
			n=kernel[0]*kernel[1]*kernel[2]*kernel[3]
			output=tf.constant([1,IMG_DISPARITY,IMG_HEIGHT,IMG_WIDTH,1])
			weights=tf.get_variable('weights',kernel,tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
			out=tf.nn.conv3d_transpose(out,weights,output,strides=stride,padding='SAME')

		return out

	"""
	def _softargmin(self,disparity):
		disparity=-tf.reshape(disparity,[IMG_DISPARITY])
		probability=tf.nn.softmax(disparity)
		d=tf.range(IMG_DISPARITY,dtype=tf.float32)
		sum=tf.reduce_sum(d*probability)/IMG_DISPARITY
		return sum
	"""
	def _todisparity(self,volume):
		with tf.variable_scope('soft_argmin'):
			d=np.arange(IMG_DISPARITY,dtype=np.float32)
			d=np.reshape(d,[IMG_DISPARITY,1,1])
			d=tf.convert_to_tensor(d)
			disparity=tf.reshape(volume,[IMG_DISPARITY,IMG_HEIGHT,IMG_WIDTH])
			probability=tf.nn.softmax(-disparity,dim=0)
			disparities=tf.reduce_sum(d*probability,0)
			disparities=tf.reshape(disparities,[1,IMG_HEIGHT,IMG_WIDTH])
		return disparities

	def _create_volume(self,linput,rinput):
		#create the cost volume from b*w*h*f to b*d*w*h*2*f
		#using split and concat to achieve the translation
		#with tf.device('/gpu:0'):
		lvolume=[]
		for i in range(int(IMG_DISPARITY/2)):
			splits=tf.split(rinput,[int(IMG_WIDTH/2-i),i],axis=2)
			rsplits=tf.concat([splits[1],splits[0]],axis=2)
			single_dis=tf.concat([linput,rsplits],3)
			lvolume.append(single_dis)
		lvol=tf.stack(lvolume,axis=0)
		rvolume=[]
		for i in range(int(IMG_DISPARITY/2)):
			splits=tf.split(linput,[i,int(IMG_WIDTH/2-i)],axis=2)
			lsplits=tf.concat([splits[1],splits[0]],axis=2)
			single_dis=tf.concat([rinput,lsplits],3)
			rvolume.append(single_dis)
		rvol=tf.stack(rvolume,axis=0)
		return lvol,rvol

	def _loss(self,pre,mode):
		ground=tf.split(self.labels,num_or_size_splits=2,axis=1)
		lground=tf.reshape(ground[mode],[1,IMG_HEIGHT,IMG_WIDTH])
		#rground=tf.reshape(ground[1],[1,IMG_HEIGHT,IMG_WIDTH])
		suml=tf.abs(pre-lground)
		loss=tf.reduce_mean(suml)
		#sumr=tf.abs(rpre-rground)
		#loss=loss+tf.reduce_mean(sumr)
		self.error1=tf.reduce_mean(tf.to_float(tf.less(suml,1)))
		#+tf.reduce_mean(tf.to_float(tf.less(sumr,1)))
		self.error2=tf.reduce_mean(tf.to_float(tf.less(suml,2)))
		#+tf.reduce_mean(tf.to_float(tf.less(sumr,2)))
		self.error3=tf.reduce_mean(tf.to_float(tf.less(suml,2)))
		#+tf.reduce_mean(tf.to_float(tf.less(sumr,2)))
		return loss
	def _build_model(self):	
		images=tf.split(self.image,num_or_size_splits=2, axis=1)
		limage=tf.reshape(images[0],[1,IMG_HEIGHT,IMG_WIDTH,3])
		rimage=tf.reshape(images[1],[1,IMG_HEIGHT,IMG_WIDTH,3])
		#residual siamese convolution
		with tf.variable_scope('siamese_conv') as scope:
			left=self.siamese_cnn(limage)
			scope.reuse_variables()
			right=self.siamese_cnn(rimage)

		#create the cost volume from w*h*f to d*w*h*(2*f)
		left,right=self._create_volume(left,right)
		left=tf.transpose(left,perm=[1,0,2,3,4])
		right=tf.transpose(right,perm=[1,0,2,3,4])
		#3d convolution
		
		with tf.variable_scope('3d_conv') as scope:
			left=self._3d_cnn(left)
			scope.reuse_variables()
			right=self._3d_cnn(right)
		
		with tf.variable_scope('soft_argmin'):
			ldisparities=self._todisparity(left)
			scope.reuse_variables()
			rdisparities=self._todisparity(right)
			#ldisparities=tf.get_variable('ldisparity',[IMG_HEIGHT,IMG_WIDTH],tf.float32,initializer=tf.random_normal_initializer())
			#rdisparities=tf.get_variable('rdisparity',[IMG_HEIGHT,IMG_WIDTH],tf.float32,initializer=tf.random_normal_initializer())
		self.lpre=tf.reshape(ldisparities,[1,IMG_HEIGHT,IMG_WIDTH,1])/255
		self.rpre=tf.reshape(rdisparities,[1,IMG_HEIGHT,IMG_WIDTH,1])/255
		side=np.random.random_integers(2)-1
		if side==0:
			self.loss=self._loss(ldisparities,side)
		else:
			self.loss=self._loss(rdisparities,side)
		tf.summary.scalar('loss',self.loss)
		tf.summary.scalar('error1',self.error1)
		tf.summary.scalar('error2',self.error2)
		tf.summary.scalar('error3',self.error3)
	def _build_train_op(self):
		"""Build training specific ops for the graph."""
		#with tf.device('/gpu:0'):
		self.lrn_rate = tf.constant(0.0001, tf.float32)
		tf.summary.scalar('learning_rate', self.lrn_rate)
		"""	
			trainable_variables = tf.trainable_variables()
			grads = gra_gpu.gradients(self.loss, trainable_variables)
			self.grad=grads
			self.var=trainable_variables
		"""
		#grads = tf.gradients(self.loss, trainable_variables)
		"""
		with tf.device('gpu:0'):
			var0=tf.trainable_variables()[:30]
			grad0=tf.gradients(self.loss, var0,name='gradients0')
		with tf.device('/gpu:1'):
			var1=tf.trainable_variables()[30:50]
			grad1=tf.gradients(self.loss, var1,name='gradients1')
		with tf.device('/gpu:2'):
			var2=tf.trainable_variables()[50:80]
			grad2=tf.gradients(self.loss, var2,name='gradients2')
		"""
		#with tf.device('/gpu:0'):
		var3=tf.trainable_variables()
		grad3=tf.gradients(self.loss, var3,name='gradients3')
		"""
		with tf.device('/gpu:3'):
			var2=tf.trainable_variables()[90:96]
			grad2=gra_gpu.gradients(self.loss, var2,name='gradients')
			"""
		#with tf.device('/gpu:0'):
		optimizer = tf.train.RMSPropOptimizer(
			self.lrn_rate,
			decay=0.9,
			momentum=0.0,
			epsilon=1e-10)
		"""
		apply_op0 = optimizer.apply_gradients(
			zip(grad0, var0),
			global_step=self.global_step, name='train_step0')
		apply_op1 = optimizer.apply_gradients(
			zip(grad1, var1),
			global_step=self.global_step, name='train_step1')
		apply_op2 = optimizer.apply_gradients(
			zip(grad2, var2),
			global_step=self.global_step, name='train_step2')
		"""
		apply_op3 = optimizer.apply_gradients(
			zip(grad3, var3),
			global_step=self.global_step, name='train_step3')
		"""
		apply_op2 = optimizer.apply_gradients(
			zip(grad2, var2),
			global_step=self.global_step, name='train_step2')
		"""
		train_ops = [apply_op3]  + self._extra_train_ops
		#train_ops = [apply_op0]+[apply_op1]+[apply_op2] + [apply_op3] + self._extra_train_ops
		self.train_op = tf.group(*train_ops)





		















