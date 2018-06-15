# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf
import config
import numpy as np
import neural_network
import os


class MODEL():

    def __init__(self):
        self.inputs = tf.placeholder(shape=[None, config.NUM_FEATURES], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, config.NUM_CLASS], dtype=tf.float32)
        self.loss = None
        self.output = None

    def build(self):
        input_data = self.inputs
        low_level_conv1 = neural_network.Convolution_layer(config.SHAPE,'low_level_conv1')
        low_level_conv2 = neural_network.Convolution_layer(config.SHAPE,'low_level_conv2')
        low_level_conv3 = neural_network.Convolution_layer(config.SHAPE,'low_level_conv3')
        low_level_conv4 = neural_network.Convolution_layer(config.SHAPE,'low_level_conv4')
        low_level_conv5 = neural_network.Convolution_layer(config.SHAPE,'low_level_conv5')
        low_level_conv6 = neural_network.Convolution_layer(config.SHAPE,'low_level_conv6')
        mid_level_conv1 = neural_network.Convolution_layer(config.SHAPE,'mid_level_conv1')
        mid_level_conv2 = neural_network.Convolution_layer(config.SHAPE,'mid_level_conv2')
        global_level_conv1 = neural_network.Convolution_layer(config.SHAPE,'global_level_conv1')
        global_level_conv2 = neural_network.Convolution_layer(config.SHAPE,'global_level_conv2')
        global_level_conv3 = neural_network.Convolution_layer(config.SHAPE,'global_level_conv3')
        global_level_conv4 = neural_network.Convolution_layer(config.SHAPE,'global_level_conv4')
        global_level_FC1 = neural_network.FullyConnected_layer(config.SHAPE,'global_level_FC1')
        global_level_FC2 = neural_network.FullyConnected_layer(config.SHAPE,'global_level_FC2')
        global_level_FC3 = neural_network.FullyConnected_layer(config.SHAPE,'global_level_FC3')
        fusion_layer = neural_network.Fusion_Layer(config.SHAPE,'fusion_layer')
        colorization_level_conv1 = neural_network.Convolution_layer(config.SHAPE,'colorization_level_conv1')
        colorization_level_conv2 = neural_network.Convolution_layer(config.SHAPE,'colorization_level_conv2')
        colorization_level_conv3 = neural_network.Convolution_layer(config.SHAPE,'colorization_level_conv3')
        colorization_level_conv4 = neural_network.Convolution_layer(config.SHAPE,'colorization_level_conv4')
        output_layer = neural_network.Output_Layer(config.SHAPE,'output_layer')

        self.output = out_layer.feed_forward(h_output)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.output))


    def train(self, data):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')

            total_batch = int(data.size/config.BATCH_SIZE)
            for epoch in range(config.NUM_EPOCHS):
                avg_cost = 0
                for batch in range(total_batch):
                    batch_X, batch_Y = data.generate_batch()
                    feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    avg_cost += loss_val / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            print("Model saved in path: %s" % save_path)

    def test(self, data):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            for i in range(len(data.dataX)):
                feed_dict = {self.inputs: [data.dataX[i]]}
                predicted = np.rint(session.run(self.output, feed_dict=feed_dict))
                print('Actual:', data.dataY[i], 'Predicted:', predicted)
