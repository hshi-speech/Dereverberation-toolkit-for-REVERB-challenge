from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn
import numpy as np

class BiLSTM(object):

    def __init__(self, sess, lr, batch_size, dim_in, dim_out, dropouts=1, training=False):
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropouts = dropouts	
        self.training = training
        self.rnn_size = 1024
        self.rnn_num_layers = 3
    
    def build(self):
        with tf.variable_scope('Neural_Network') as vs:

            def lstm_cell():
               return tf.contrib.rnn.LSTMCell(
                  self.rnn_size, forget_bias=1.0, use_peepholes=True,
                  initializer=tf.contrib.layers.xavier_initializer(),
                  state_is_tuple=True, activation=tf.tanh)
            attn_cell = lstm_cell
            if self.training and self.dropouts < 1.0:
               def attn_cell():
                   return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.dropouts)

            with tf.variable_scope('Intputs'):
                self.x_noisy = tf.placeholder(tf.float32, shape=[None, self.dim_in[0], self.dim_in[1]], name='x')
				
            with tf.variable_scope('Outputs'):
                self.y_clean = tf.placeholder(tf.float32, shape=[None, self.dim_out], name='y_clean')
                # self.y_clean = tf.reshape(self.y_clean, (-1, self.dim_out))
									 
            with tf.variable_scope('DNN'):
                 inputs = tf.reshape(self.x_noisy, (-1, self.dim_in[0]*self.dim_in[1]))
                 layer1 = tf.layers.dense(inputs=inputs, units=1024, activation=tf.nn.relu)
                 # layer1 = tf.layers.dropout(layer1, rate=self.dropouts, training=self.training)
                 layer1 = tf.reshape(layer1, [-1, 1, 1024])

                 lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.rnn_num_layers)],state_is_tuple=True)
                 lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.rnn_num_layers)],state_is_tuple=True)

                 lstm_fw_cell = _unpack_cell(lstm_fw_cell)
                 lstm_bw_cell = _unpack_cell(lstm_bw_cell)
                 result = rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw = lstm_fw_cell,
                    cells_bw = lstm_bw_cell,
                    inputs=layer1,
                    dtype=tf.float32)
                    # sequence_length=self.batch_size)
                 layer2, fw_final_states, bw_final_states = result

                 layer2 = tf.reshape(layer2, [-1, 2*self.rnn_size])
                 in_size=2*self.rnn_size

                 self.enhanced_outputs = tf.layers.dense(inputs=layer2 , units=self.dim_out, activation=None)


            with tf.name_scope('loss'):
                 self.loss = tf.losses.mean_squared_error(self.y_clean, self.enhanced_outputs)
                 tf.summary.scalar('Loss', self.loss)
				
            with tf.name_scope("exp_learning_rate"):
                self.global_step = tf.Variable(0, trainable=False)
                self.exp_learning_rate = tf.train.exponential_decay(self.lr,
                                                             global_step=self.global_step,
                                                             decay_steps=50000, decay_rate=0.8, staircase=False)
                tf.summary.scalar('Learning rate', self.exp_learning_rate)
				
            optimizer = tf.train.AdamOptimizer(self.lr)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            self.optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                      global_step=self.global_step)
            self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)				

    def outputs(tr_x):
         self.x_noisy = tr_x

def _unpack_cell(cell):
    if isinstance(cell,tf.contrib.rnn.MultiRNNCell):
        return cell._cells
    else:
        return [cell]

