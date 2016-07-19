# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:03:52 2016

@author: whr94621
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

##################
# Data Generator #
##################

def data_generator(f_x, f_y, block_bytes=3 * 50 * 8):
    """Generate data from binary file by bytes

    Args:
        f_x: string, path of traininig data.
        f_y: string, path of the labels of training  data.
        block_bytes: integer, the size of one batch training data.

    yield:
        v: numpy ndarray, training data
        label: the label of training data
    """

    with open(f_x, 'rb') as f1, open(f_y, 'r') as f2:
        v = np.fromstring(f1.read(block_bytes))
        label = f2.readline().strip()
        while v.shape[0] and label:
            label = float(label)
            yield (v, label)

            v = np.fromstring(f1.read(block_bytes))
            label = f2.readline().strip()

##########
# Metric #
##########
def metric(x, y):
    """Metric to evaluate model

    This function is a wrapper of the metrics module in skitlearn. Given
    predicted labels and test labels, this function could return precision,
    recall and F1 score simultaneously.
    """

    x[x > 0] = 1
    x[x < 0] = -1
    precison = precision_score(y, x)
    recall = recall_score(y, x)
    f1 = f1_score(y, x)

    return precison, recall, f1
###############
# Initilizers #
###############
def _matrix_initializer(shape, name, stddev=0.5):
    """Initialize Matrix

    Given shape, this function initialize the matrix by gaussian initializer
    with stddev as its variance.

    Args:
        shape: python list, specify the shape the matrix
        name: string, the name in tensorflow graph
        stddev: float, the variance of the gaussian initializer

    """
    return tf.Variable(tf.random_normal(shape, stddev=stddev), name=name, dtype=tf.float32)

def _vector_initializer(shape, name):
    """Initialize vector

    Given shape, this function initialize a vector
    Args:
        shape: int, specify the length of the matrix
        name: string, the name in tensorflow graph
    """
    return tf.Variable(tf.zeros(shape), name=name, dtype=tf.float32)

##################
# Layers Builder #
##################
class _feature_expansion_layer(object):
    """Build a layer to expansion feature

    This layer is a just a hidden layer without bias.For example, the input
    shape is [None, 50, 3], which means each data has 3 50-dimensions
    feature vectors.
    We build a feature expansion layer with a 3*20 matrix, and the output
    will be [None, 50, 3] * [3, 20] = [None, 50, 20]. By learning the variables
    of this layer, we will finally expansion the feature of the data from 3
    to 20.
    """
    def __init__(self, shape, name, activation=tf.tanh):
        self._W = _matrix_initializer(shape, name+'_W')
        self._activation = activation
        self._shape = shape
        self.param = [self._W]

    def output(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [-1, self._shape[0]])
        linear = tf.matmul(x, self._W)
        return self._activation(tf.reshape(linear, [batch_size, -1]))


class _hidden_layer_with_bias(object):
    def __init__(self, shape, name, activation=tf.tanh):
        self._W = _matrix_initializer(shape, name+'_W')
        self._b = _vector_initializer([shape[1]], name+'_b')
        self._activation = activation
        self.param = [self._W, self._b]

    def output(self, x):
        linear = tf.matmul(x, self._W) + self._b
        return self._activation(linear)

##################
# Graph Builder #
#################

class MLPwL:
    def __init__(self, expand_layer=[3,20], layers=[1000, 250, 50, 1],
                 input_shape=[None, 50, 3], learning_rate=0.1, momentum=0.6):

        self.layers = []
        self.layers.append(_feature_expansion_layer(expand_layer,
                                                 '_feature_expansion_layer',
                                                 tf.tanh))
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            self.layers.append(
                _hidden_layer_with_bias([n_in, n_out],
                                    'HiddenLayer_' + str(n_in) +str(n_out),
                                        tf.tanh))
        self.global_step = tf.Variable(0, trainable=False)
        self.x_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=input_shape)

        self.y_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])

        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   self.global_step,
                                                   100000, 0.96,
                                                   staircase=True)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        x = tf.reshape(x, [-1])
        return x

    def loss_function(self, y, y_true):
        y_true = tf.to_float(y_true)
        error = y - y_true
        error = error * tf.to_float(error * y_true < 0)
        return tf.reduce_mean(error * error)

    def train_op(self, loss):
        #loss = self.loss_function(x, y_true)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return train_op

class MLP:
    '''
    Class for Buildinig Multi Layer Preceptron
    '''
    def __init__(self, layers=[150, 50, 1], input_shape=[None, 150],
                 learning_rate=0.1,momentum=0.8):
        self.layers = []
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            self.layers.append(
                _hidden_layer_with_bias([n_in, n_out],
                                    'HiddenLayer_' + str(n_in) +str(n_out),
                                        tf.tanh))

        self.x_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=input_shape)

        self.y_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   self.global_step,
                                                   100000, 0.96,staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                beta1=momentum,
                                                beta2=0.999, epsilon=1e-08)

    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        x = tf.reshape(x, [-1])
        return x

    def loss_function(self, y, y_true):
        y_true = tf.to_float(y_true)
        error = y - y_true
        error = error * tf.to_float(error * y_true < 0)
        return tf.reduce_mean(error * error)

    def train_op(self, loss):
        #loss = self.loss_function(x, y_true)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return train_op


