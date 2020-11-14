from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import tensorflow as tf
import numpy as np

class DNN(object):

    def __init__(self, sess, lr, batch_size, dim_in, dim_out, dropouts=0, training=False):
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropouts = dropouts	
        self.training = training
    
    def build(self):
        with tf.variable_scope('Neural_Network') as vs:
            with tf.variable_scope('Intputs'):
                self.x_noisy = tf.placeholder(tf.float32, shape=[None, self.dim_in[0], self.dim_in[1]], name='x')
				
            with tf.variable_scope('Outputs'):
                self.y_clean = tf.placeholder(tf.float32, shape=[None, self.dim_out], name='y_clean')
                # self.y_clean = tf.reshape(self.y_clean, (-1, self.dim_out))
									 
            with tf.variable_scope('DNN'):
                 inputs = tf.reshape(self.x_noisy, (-1, self.dim_in[0]*self.dim_in[1]))
                 layer1 = tf.layers.dense(inputs=inputs, units=1024, activation=tf.nn.relu)
                 layer1 = tf.layers.dropout(layer1, rate=self.dropouts, training=self.training)
                 layer2 = tf.layers.dense(inputs=layer1, units=1024, activation=tf.nn.relu)
                 layer2 = tf.layers.dropout(layer2, rate=self.dropouts, training=self.training)		
                 layer3 = tf.layers.dense(inputs=layer2, units=1024, activation=tf.nn.relu)
                 layer3 = tf.layers.dropout(layer3, rate=self.dropouts, training=self.training)
                 self.enhanced_outputs = tf.layers.dense(inputs=layer3 , units=self.dim_out, activation=None)
                 # self.enhanced_outputs = tf.reshape(self.enhanced_outputs, (-1, self.dim_out))

            with tf.name_scope('loss'):
                 self.loss = tf.losses.mean_squared_error(self.y_clean, self.enhanced_outputs)
                 tf.summary.scalar('Loss', self.loss)
				
            with tf.name_scope("exp_learning_rate"):
                self.global_step = tf.Variable(0, trainable=False)
                self.exp_learning_rate = tf.train.exponential_decay(self.lr,
                                                             global_step=self.global_step,
                                                             decay_steps=500000, decay_rate=0.95, staircase=False)
                tf.summary.scalar('Learning rate', self.exp_learning_rate)
				
            optimizer = tf.train.AdamOptimizer(self.lr)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            self.optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                      global_step=self.global_step)
            self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)				

    def outputs(tr_x):
         self.x_noisy = tr_x



