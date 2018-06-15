# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf


class Layer():

    def __init__(self, shape, name):
        with tf.variable.scope(name) as scope:
            self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=5e-2))
            self.biases = tf.Variable(tf.constant(0.0, shape=shape[-1]))

    def feed_forward(self, input_data, stride=None):
        raise NotImplementedError


class Convolution_Layer(Layer):

    def __init__(self, shape, name):
        with tf.variable.scope(name) as scope:
            super(Convolution_Layer, self).__init__(shape, name)

    def feed_forward(self, input_data, stride):
        conv = tf.nn.conv2d(input_data, self.weights, stride, padding="SAME")
        output_data = tf.nn.relu(tf.nn.bias_add(conv, self.biases))
        return output_data


class FullyConnected_layer(Layer):

    def __init__(self, shape, name):
        with tf.variable.scope(name) as scope:
            super(FullyConnected_layer, self).__init__(shape, name)

    def feed_forward():
        fullyconnected = tf.matmul(global_level_FC1, self.weights)
        output_data = tf.nn.relu(tf.nn.bias_add(conv, self.biases))


class Fusion_Layer(Convolution_Layer):

    def __init__(self, shape, name):
        super(Fusion_Layer, self).__init__(shape, name)

    def feed_forward(self, mid_features, global_features, stride):
        mid_features_reshaped = tf.reshape(mid_features,[-1,256])
        mid_features_reshaped = tf.unstack(mid_level_conv2_reshaped,axis=0)
        fusion_level = [tf.concat([see_mid, global_features],0) for see_mid in mid_features_reshaped]
        fusion_level = tf.stack(fusion_level)
        fusion_level = tf.shape(fusion_level,[28,28,512])
        return super(Fusion_Layer, self).feed_forward(fusion_level, stride)


class Output_Layer(Layer):

    def __init__(self, shape, name):
        super(Output_Layer, self).__init__(shape, name)

    def feed_forward(self, input_data, stride):
        conv = tf.nn.conv2d(input_data, self.weights, stride, padding='SAME')
        output_data = tf.nn.sigmoid(tf.nn.bias_add(conv, self.biases))
        return output_data
