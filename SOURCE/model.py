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


class MODEL():

    def __init__(self):
        self.inputs = tf.placeholder(shape=[config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 1], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 2], dtype=tf.float32)
        self.loss = None
        self.output = None

    def build(self):
        input_data = self.inputs

        low_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 1, 64], stddev=0.1, value=0.1)
        h = low_level_conv1.feed_forward(input_data=input_data, stride=[1, 2, 2, 1])

        low_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 64, 128],stddev=0.1, value=0.1)
        h = low_level_conv2.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        low_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 128, 128], stddev=0.1, value=0.1)
        h = low_level_conv3.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        low_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 128, 256], stddev=0.1, value=0.1)
        h = low_level_conv4.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        low_level_conv5 = neural_network.Convolution_Layer(shape=[3, 3, 256, 256], stddev=0.1, value=0.1)
        h = low_level_conv5.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        low_level_conv6 = neural_network.Convolution_Layer(shape=[3, 3, 256, 512], stddev=0.1, value=0.1)
        h = low_level_conv6.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        mid_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h1 = mid_level_conv1.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        mid_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 512, 256], stddev=0.1, value=0.1)
        h1 = mid_level_conv2.feed_forward(input_data=h1, stride=[1, 1, 1, 1])

        global_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h2 = global_level_conv1.feed_forward(input_data=h, stride=[1, 2, 2, 1])

        global_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h2 = global_level_conv2.feed_forward(input_data=h2, stride=[1, 1, 1, 1])

        global_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h2 = global_level_conv3.feed_forward(input_data=h2, stride=[1, 2, 2, 1])

        global_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        h2 = global_level_conv4.feed_forward(input_data=h2, stride=[1, 1, 1, 1])

        h2_flat = tf.reshape(h2, [config.BATCH_SIZE,-1])
        dim = h2_flat.get_shape()[1].value
        global_level_FC1 = neural_network.FullyConnected_Layer(shape=[dim, 1024], stddev=0.04, value=0.1)
        h2 = global_level_FC1.feed_forward(input_data=h2_flat)

        global_level_FC2 = neural_network.FullyConnected_Layer(shape=[1024, 512], stddev=0.04, value=0.1)
        h2 = global_level_FC2.feed_forward(input_data=h2)

        global_level_FC3 = neural_network.FullyConnected_Layer(shape=[512, 256], stddev=0.04, value=0.1)
        h2 = global_level_FC3.feed_forward(input_data=h2)

        fusion_layer = neural_network.Fusion_Layer(shape=[1, 1, 512, 256], stddev=0.1, value=0.1)
        h = fusion_layer.feed_forward(h1, h2, stride=[1, 1, 1, 1])

        colorization_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 256, 128], stddev=0.1, value=0.1)
        h = colorization_level_conv1.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        h = tf.image.resize_images(h, [56, 56], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        colorization_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 128, 64], stddev=0.1, value=0.1)
        h = colorization_level_conv2.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        colorization_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 64, 64], stddev=0.1, value=0.1)
        h = colorization_level_conv3.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        h = tf.image.resize_images(h, [112, 112], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        colorization_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 64, 32], stddev=0.1, value=0.1)
        h = colorization_level_conv4.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        output_layer = neural_network.Output_Layer(shape=[3, 3, 32, 2], stddev=0.1, value=0.1)
        logits = output_layer.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        logits_norm = tf.image.convert_image_dtype(logits, tf.float32)
        self.output = tf.image.resize_images(logits_norm, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.output))

    def train(self, data):
#        global_step = tf.Variable(0, trainable=False)
#        learning_rate = tf.train.exponential_decay(0.1, global_step, data.size, 0.95, staircase=True)
#        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-08).minimize(self.loss, global_step)
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')

            total_batch = int(data.size/config.BATCH_SIZE)
            for epoch in range(config.NUM_EPOCHS):
                avg_cost = 0
                for batch in range(total_batch):
                    batch_X, batch_Y, _ = data.generate_batch()
                    feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    avg_cost += loss_val / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            print("Model saved in path: %s" % save_path)

    def deprocess(self, imgs):
        imgs = imgs * 255
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0
        return imgs.astype(np.uint8)

    def reconstruct(self, batch_X, predicted_Y, filelist):
        for i in range(config.BATCH_SIZE):
            result = np.concatenate((batch_X[i], predicted_Y[i]), axis=2)
            result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
            save_path = os.path.join(config.OUT_DIR, filelist[i][:-4] + "reconstructed.jpg")
            cv2.imwrite(save_path, result) 
            
    def test(self, data):
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            avg_cost = 0
            total_batch = int(data.size/config.BATCH_SIZE)
            for batch in range(total_batch):
                batch_X, batch_Y, filelist = data.generate_batch()
                feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                pred_Y, loss = session.run([self.output, self.loss], feed_dict=feed_dict)
                self.reconstruct(self.deprocess(batch_X), self.deprocess(pred_Y), filelist)
                avg_cost += loss/total_batch
            print("cost =", "{:.3f}".format(avg_cost))
