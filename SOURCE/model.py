# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf
import config
import neural_network
import os
import numpy as np
import cv2
from scipy.misc import imsave 

class MODEL():

    def __init__(self):
        self.inputs = tf.placeholder(shape=[config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 1], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 2], dtype=tf.float32)
        self.loss = None
        self.output = None

    def build(self):
        input_data = self.inputs

        low_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 1, 64], stddev=5e-2, value=0.0)
        h = low_level_conv1.feed_forward(input_data=input_data, stride=[1, 2, 2, 1])

        low_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 64, 128],stddev=5e-2, value=0.0)
        h = low_level_conv2.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        low_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 128, 128], stddev=5e-2, value=0.0)
        h = low_level_conv3.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        low_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 128, 256], stddev=5e-2, value=0.0)
        h = low_level_conv4.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        low_level_conv5 = neural_network.Convolution_Layer(shape=[3, 3, 256, 256], stddev=5e-2, value=0.0)
        h = low_level_conv5.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        low_level_conv6 = neural_network.Convolution_Layer(shape=[3, 3, 256, 512], stddev=5e-2, value=0.0)
        h = low_level_conv6.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        mid_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=5e-2, value=0.0)
        h1 = mid_level_conv1.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        mid_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 512, 256], stddev=5e-2, value=0.0)
        h1 = mid_level_conv2.feed_forward(input_data=h1, stride=[1, 1, 1, 1])

        global_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=5e-2, value=0.0)
        h2 = global_level_conv1.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        global_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=5e-2, value=0.0)
        h2 = global_level_conv2.feed_forward(input_data=h2, stride=[1, 1, 1, 1])

        global_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=5e-2, value=0.0)
        h2 = global_level_conv3.feed_forward(input_data=h2, stride=[1, 2, 2, 1])

        global_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=5e-2, value=0.0)
        h2 = global_level_conv4.feed_forward(input_data=h2, stride=[1, 1, 1, 1])

        reshape = tf.reshape(h2, [config.BATCH_SIZE,-1])
        dim = reshape.get_shape()[1].value
        global_level_FC1 = neural_network.FullyConnected_Layer(shape=[dim, 1024], stddev=0.04, value=0.1)
        h2 = global_level_FC1.feed_forward(input_data=reshape)

        global_level_FC2 = neural_network.FullyConnected_Layer(shape=[1024, 512], stddev=0.04, value=0.1)
        h2 = global_level_FC2.feed_forward(input_data=h2)

        global_level_FC3 = neural_network.FullyConnected_Layer(shape=[512, 256], stddev=0.04, value=0.1)
        h2 = global_level_FC3.feed_forward(input_data=h2)

        fusion_layer = neural_network.Fusion_Layer(shape=[1, 1, 512, 256], stddev=5e-2, value=0.0)
        h = fusion_layer.feed_forward(h1, h2, stride=[1, 1, 1, 1])

        colorization_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 256, 128], stddev=5e-2, value=0.0)
        h = colorization_level_conv1.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        colorization_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 128, 64], stddev=5e-2, value=0.0)
        h = colorization_level_conv2.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        colorization_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 64, 64], stddev=5e-2, value=0.0)
        h = colorization_level_conv3.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        colorization_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 64, 32], stddev=5e-2, value=0.0)
        h = colorization_level_conv4.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        output_layer = neural_network.Output_Layer(shape=[3, 3, 32, 2], stddev=5e-2, value=0.0)
        logits = output_layer.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        logits_norm = tf.image.convert_image_dtype(logits, tf.float32)/255.
        self.output = tf.image.resize_images(logits_norm, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.output))

    def train(self, data):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.01, global_step, data.size, 0.95, staircase=True)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-08).minimize(self.loss, global_step)
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

    def deprocess(imgs):
        imgs = imgs * 255
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0
        return imgs
    
    def reconstruct(batch_X, predicted_Y):
        global count
        for i in range(config.BATCH_SIZE):
            result = np.zeros([config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
            result[:, :, 0] = batch_X[i]
            result[:, :, 1:3] = predicted_Y[i]
            result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
            save_path = os.path.join(config.RESULT, "Img" + str(count))
            count += 1
            imsave(save_path, result)
            
    def test(self, data):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            avg_cost = 0
            total_batch = int(data.size/config.BATCH_SIZE)
            for batch in range(total_batch):
                batch_X, batch_Y = data.generate_batch()
                feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                pred_Y, loss = session.run([self.output, self.loss], feed_dict=feed_dict)
                pred_Y = self.deprocess(pred_Y)
                self.reconstruct(batch_X, pred_Y)
                avg_cost += loss/total_batch
            print("cost =", "{:.3f}".format(avg_cost))
