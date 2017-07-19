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
IMG_WIDTH = 512 
IMG_HEIGHT = 256
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

def train():
	#with tf.device('/cpu:0'):
	images,disparities,name=get_input(1) 
	#tf.device('/gpu:0')
	#get input data
	model=whole_model.E2EModel(images,disparities,'train')
	model.build_graph()
	images_s=tf.split(images,num_or_size_splits=2,axis=1)
	limg_s=tf.reshape(images_s[0],[1,IMG_HEIGHT,IMG_WIDTH,3])
	rimg_s=tf.reshape(images_s[1],[1,IMG_HEIGHT,IMG_WIDTH,3])
	ground=tf.split(disparities,num_or_size_splits=2,axis=1)
	lground=tf.reshape(ground[0],[1,IMG_HEIGHT,IMG_WIDTH,1])
	rground=tf.reshape(ground[1],[1,IMG_HEIGHT,IMG_WIDTH,1])
	summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=r'D:\GC-Base\log\output',
      summary_op=tf.summary.merge([model.summaries,
      							tf.summary.image('oril',lground/255,max_outputs=1),tf.summary.image('orir',rground/255,max_outputs=1),
      							tf.summary.image('limg_s',limg_s,max_outputs=3),tf.summary.image('rimg_s',rimg_s,max_outputs=3),
      							tf.summary.image('lpre',model.lpre,max_outputs=1),tf.summary.image('rpre',model.lpre,max_outputs=1)]))
	logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.loss,
               'error1': model.error1,
               'error2': model.error2,
               'error3':model.error3
               },
      every_n_iter=100)
	class _LearningRateSetterHook(tf.train.SessionRunHook):
		"""Sets learning_rate based on global step."""

		def begin(self):

			self._lrn_rate = 0.0001

		def before_run(self, run_context):
			return tf.train.SessionRunArgs(
			model.global_step,  # Asks for global step value.
			feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

		def after_run(self, run_context, run_values):
			train_step = run_values.results
			if train_step < 40000:
				self._lrn_rate = 0.0001
			elif train_step < 60000:
				self._lrn_rate = 0.0001
			elif train_step < 80000:
				self._lrn_rate = 0.0001
			else:
				self._lrn_rate = 0.0001
	with tf.train.MonitoredTrainingSession(
      checkpoint_dir=r'D:\GC-Base\log',
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=None,
      config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as mon_sess:
		#print('running'+str(model.global_step))
		while not mon_sess.should_stop():
			mon_sess.run(model.train_op)
			steps=model.global_step.eval(session=mon_sess)
			print('running'+str(steps))
			if steps%100==0:
				print('Now image comes to:'+str(name.eval(session=mon_sess).decode('UTF-8')))
			"""
			if setps>1 and model.save==1:
				b_summary_op=tf.summary.merge([model.summaries,
      							tf.summary.image('lpre',model.lpre,max_outputs=1),tf.summary.image('rpre',model.lpre,max_outputs=1)])
				saver = tf.train.Saver(b_summary_op)	
				saver.save(mon_sess,'best_model',global_step=steps)
			"""
			"""
			print('model.var',len(model.var))
			print('model.grad',len(model.grad))

			if str(model.global_step.eval(session=mon_sess))=='0' or str(model.global_step.eval(session=mon_sess))==0:
				outputlog=open(r'D:\GC-Base\logs.txt','w+')
				for i in range(len(model.var)):
					outputlog.write(str(i)+'\n')
					outputlog.write(model.var[i].name+'\n')
					outputlog.write(model.grad[i].name+'\n')
				outputlog.close()
			"""
			#b=model.grads
			#print(len(b))
			#print(model.grads[1].eval(session=mon_sess))
	"""
	#E2ENet=E2EModel(images,disparities)
	init_op = tf.global_variables_initializer()
	with tf.Session() as sess:
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)
	    image=images.eval()
	    disparitiy=disparities.eval()
	    coord.request_stop()
	    coord.join(threads)
	"""
def evaluate():
	
	"""
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
train()
"""
for d in ['/gpu:0','/gpu:1','/gpu:2', '/gpu:3']:
	with tf.device(d):
		train()

	 
 
 
 
 
 
 
 