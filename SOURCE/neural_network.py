# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf
import config


class Layer():

    def __init__(self, shape, stddev, value):
        self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
        self.biases = tf.Variable(tf.constant(value=value, shape=[shape[-1]]))

    def feed_forward(self, input_data, stride=None):
        raise NotImplementedError


class Convolution_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(Convolution_Layer, self).__init__(shape, stddev, value)

    def feed_forward(self, input_data, stride):
        conv = tf.nn.conv2d(input_data, self.weights, stride, padding="SAME")
        output_data = tf.nn.tanh(tf.nn.bias_add(conv, self.biases))
        return output_data


class FullyConnected_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(FullyConnected_Layer, self).__init__(shape, stddev, value)

    def feed_forward(self, input_data, stride=None):
        fullyconnected = tf.matmul(input_data, self.weights)
        output_data = tf.nn.relu(tf.nn.bias_add(fullyconnected, self.biases))
        return output_data


class Fusion_Layer(Convolution_Layer):

    def __init__(self, shape, stddev, value):
        super(Fusion_Layer, self).__init__(shape, stddev, value)

    def feed_forward(self, mid_features, global_features, stride):
        mid_features_shape = mid_features.get_shape().as_list()
        mid_features_reshaped = tf.reshape(mid_features, [config.BATCH_SIZE, mid_features_shape[1]*mid_features_shape[2], 256])
        fusion_level = []
        for j in range(mid_features_reshaped.shape[0]):
            for i in range(mid_features_reshaped.shape[1]):
                see_mid = mid_features_reshaped[j, i, :]
                see_mid_shape = see_mid.get_shape().as_list()
                see_mid = tf.reshape(see_mid, [1, see_mid_shape[0]])
                global_features_shape = global_features[j, :].get_shape().as_list()
                see_global = tf.reshape(global_features[j, :], [1, global_features_shape[0]])
                fusion = tf.concat([see_mid, see_global], 1)
                fusion_level.append(fusion)
        fusion_level = tf.stack(fusion_level, 1)
        fusion_level = tf.reshape(fusion_level, [config.BATCH_SIZE, 28, 28, 512])
        return super(Fusion_Layer, self).feed_forward(fusion_level, stride)


class Output_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(Output_Layer, self).__init__(shape, stddev, value)

    def feed_forward(self, input_data, stride):
        conv = tf.nn.conv2d(input_data, self.weights, stride, padding='SAME')
        output_data = tf.nn.sigmoid(tf.nn.bias_add(conv, self.biases))
        return output_data
