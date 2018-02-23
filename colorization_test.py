from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import re
import sys
import tarfile
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

IMAGE_SIZE = 224
NUM_CLASSES = 10
NUM_EPOCHS = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
EVAL_FREQUENCY = 1
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 2,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('batch_size_test', 1,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

TOWER_NAME = 'tower'

def data_type():
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


def error_rate(predictions, batch_labels):
    err_rate = 0
    batch_labels_shape = np.shape(batch_labels)
    for i in range(batch_labels_shape[0]):
        err_rate += mse(predictions[i], batch_labels[i])        
    return err_rate

def display_images(color_imgs):
    color_imgs_shape = np.shape(color_imgs)
    for i in range(color_imgs_shape[0]):
       # print("i is " + str(i))
        cv2.imshow('image',color_imgs[i])
        cv2.waitKey(0) 
    return 
    
def read_img(filename):
    img = cv2.imread(filename)
    height, width, channels = img.shape
   # greyimg = cv2.cvtColor(cv2.resize(img, (224, 224)),cv2.COLOR_BGR2GRAY)
    colorimg = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)),cv2.COLOR_BGR2LAB)
    #return greyimg, colorimg
    return colorimg

def create(dirname):
    filenames = os.listdir(dirname)
    print("dirname " + str(dirname))
    x_img = []
    y_img = []
    for filename in filenames:
        #grey, color = read_img(dirname+filename)
        #grey_3 = np.atleast_3d(grey)
        color = read_img(dirname + filename)
        color_l = color[:,:,0]
        color_l3 = np.atleast_3d(color_l)
        color_ab = color[:,:,1:]
        x_img.append(color_l3)
        y_img.append(color_ab)
    return x_img, y_img
def _activation_summary(x):
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))
    
def create_network(image):
	# low_level_conv1
    with tf.variable_scope('low_level_conv1') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 64], stddev=5e-2))
        bias = tf.Variable(tf.constant(0.0, shape=[64]))
        conv = tf.nn.conv2d(image, weight, [1, 2, 2, 1], padding="SAME")
        low_level_conv1 = tf.nn.relu(tf.nn.bias_add(conv, bias))
        _activation_summary(low_level_conv1)

	# low_level_conv2
    with tf.variable_scope('low_level_conv2') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128],stddev=5e-2))
        bias = tf.Variable(tf.constant(0.0, shape=[128]))
        conv = tf.nn.conv2d(low_level_conv1, weight, [1, 1, 1, 1], padding="SAME")
        low_level_conv2 = tf.nn.relu(tf.nn.bias_add(conv, bias))
        _activation_summary(low_level_conv2)
    
	# low_level_conv3
    with tf.variable_scope('low_level_conv3') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        biases = tf.Variable(tf.constant(0.0, shape=[128]))
        conv = tf.nn.conv2d(low_level_conv2, weight, [1, 2, 2, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv3)

	# low_level_conv4
    with tf.variable_scope('low_level_conv4') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))
        biases = tf.Variable(tf.constant(0.0, shape=[256]))
        conv = tf.nn.conv2d(low_level_conv3, weight, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv4)    

	# low_level_conv5
    with tf.variable_scope('low_level_conv5') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))
        conv = tf.nn.conv2d(low_level_conv4, weight, [1, 2, 2, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256]))
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv5 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv5)

	# low_level_conv6
    with tf.variable_scope('low_level_conv6') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=5e-2))
        conv = tf.nn.conv2d(low_level_conv5, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512]))
        pre_activation = tf.nn.bias_add(conv, biases)
        low_level_conv6 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(low_level_conv6)

	# mid_level_conv1
    with tf.variable_scope('mid_level_conv1') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512],stddev=5e-2))
        conv = tf.nn.conv2d(low_level_conv6, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512]))
        pre_activation = tf.nn.bias_add(conv, biases)
        mid_level_conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(mid_level_conv1)

	# mid_level_conv2
    with tf.variable_scope('mid_level_conv2') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 256], stddev=5e-2))
        conv = tf.nn.conv2d(mid_level_conv1, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256]))
        pre_activation = tf.nn.bias_add(conv, biases)
        mid_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(mid_level_conv2)

	# global_level_conv1
    with tf.variable_scope('global_level_conv1') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2))
        conv = tf.nn.conv2d(low_level_conv6, weight, [1, 2, 2, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512]))
        pre_activation = tf.nn.bias_add(conv, biases)
        global_level_conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(global_level_conv1)

	# global_level_conv2
    with tf.variable_scope('global_level_conv2') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2))
        conv = tf.nn.conv2d(global_level_conv1, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512]))
        pre_activation = tf.nn.bias_add(conv, biases)
        global_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(global_level_conv2)

	# global_level_conv3
    with tf.variable_scope('global_level_conv3') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2))
        conv = tf.nn.conv2d(global_level_conv2, weight, [1, 2, 2, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512]))
        pre_activation = tf.nn.bias_add(conv, biases)
        global_level_conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(global_level_conv3)

	# global_level_conv4
    with tf.variable_scope('global_level_conv4') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2))
        conv = tf.nn.conv2d(global_level_conv3, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512]))
        pre_activation = tf.nn.bias_add(conv, biases)
        global_level_conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(global_level_conv4)
        global_level_conv4_shape = global_level_conv4.get_shape().as_list()
        
	# global_level_FC1
    with tf.variable_scope('global_level_FC1') as scope:
        #flatten = tf.reshape(conv_5, [conv5_shape[0], conv5_shape[1]*conv5_shape[2]*conv5_shape[3]])
        #print("global level conv4 shape " + str(global_level_conv4_shape))
        flatten = tf.reshape(global_level_conv4, [FLAGS.batch_size, global_level_conv4_shape[1] * global_level_conv4_shape[2] * global_level_conv4_shape[3]])
        #print("reshape global level conv4 " + str(flatten.get_shape()))
        dim = flatten.get_shape()[1].value
        weight = tf.Variable(tf.truncated_normal(shape=[dim, 1024], stddev=0.04))
        biases = tf.Variable(tf.constant(0.1, shape=[1024]))
        global_level_FC1 = tf.nn.relu(tf.matmul(flatten, weight) + biases, name=scope.name)
        _activation_summary(global_level_FC1)
        print("global level FC1 " + str(global_level_FC1.get_shape()))
	
	# global_level_FC2
    with tf.variable_scope('global_level_FC2') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.04))
        biases = tf.Variable(tf.constant(0.1, shape=[512]))
        global_level_FC2 = tf.nn.relu(tf.matmul(global_level_FC1, weight) + biases, name=scope.name)
        _activation_summary(global_level_FC2)
 #       print("global level FC2 " + str(global_level_FC2.get_shape()))
    
	# global_level_FC3
    with tf.variable_scope('global_level_FC3') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[512, 256], stddev=0.04))
        biases = tf.Variable(tf.constant(0.1, shape=[256]))
        global_level_FC3 = tf.nn.relu(tf.matmul(global_level_FC2, weight) + biases, name=scope.name)
        _activation_summary(global_level_FC3)
        print("global level FC3 " + str(global_level_FC3.get_shape()))
        
	# fusion_layer
    with tf.variable_scope('fusion_layer') as scope:
        print("mid level conv2 shape " + str(mid_level_conv2.get_shape()))
        mid_level_conv2_shape = mid_level_conv2.get_shape().as_list()
        mid_level_conv2_reshaped = tf.reshape(mid_level_conv2, [FLAGS.batch_size, mid_level_conv2_shape[1] * mid_level_conv2_shape[2], 256])
        fusion_level = []
        for j in range(mid_level_conv2_reshaped.shape[0]):
            for i in range(mid_level_conv2_reshaped.shape[1]):
                see_mid = mid_level_conv2_reshaped[j, i, :]
                see_mid_shape = see_mid.get_shape().as_list()
                see_mid = tf.reshape(see_mid, [1, see_mid_shape[0]])
                global_level_FC3_shape = global_level_FC3[j, :].get_shape().as_list()
                see_global = tf.reshape(global_level_FC3[j, :], [1, global_level_FC3_shape[0]])
                fusion = tf.concat([see_mid, see_global], 1)
                fusion_level.append(fusion)
#        print("1 . mid level conv2 reshape " + str(mid_level_conv2_reshaped.get_shape()))
        #mid_level_conv2_reshaped = tf.unstack(mid_level_conv2_reshaped,axis=0)	
        #mp = tf.stack([mid_level_conv2_reshaped])
                
                #print("fusion " + str(fusion.get_shape()))
            
        fusion_level = tf.stack(fusion_level, 1)
        print("fusion level " + str(fusion_level.get_shape()))
        #fusion_level = [tf.concat([see_mid,global_level_FC3],0) for see_mid in mid_level_conv2_reshaped]
        #fusion_level = tf.stack(fusion_level)
        fusion_level = tf.reshape(fusion_level,[FLAGS.batch_size, 28, 28, 512])	
        weight = tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 256], stddev=5e-2))
        conv = tf.nn.conv2d(fusion_level, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256]))
        pre_activation = tf.nn.bias_add(conv, biases)
        fusion_layer = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(fusion_layer)
	
	# colorization_level_conv1
    with tf.variable_scope('colorization_level_conv1') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 128], stddev=5e-2))
        conv = tf.nn.conv2d(fusion_layer, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128]))
        pre_activation = tf.nn.bias_add(conv, biases)
        colorization_level_conv1 = tf.nn.relu(pre_activation, name=scope.name)
        colorization_level_conv1_upsampled = tf.image.resize_images(colorization_level_conv1, [56, 56], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        _activation_summary(colorization_level_conv1_upsampled)
		
	# colorization_level_conv2
    with tf.variable_scope('colorization_level_conv2') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 64], stddev=5e-2))
        conv = tf.nn.conv2d(colorization_level_conv1_upsampled, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64]))
        pre_activation = tf.nn.bias_add(conv, biases)
        colorization_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(colorization_level_conv2)

	# colorization_level_conv3
    with tf.variable_scope('colorization_level_conv3') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))
        conv = tf.nn.conv2d(colorization_level_conv2, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64]))
        pre_activation = tf.nn.bias_add(conv, biases)
        colorization_level_conv3 = tf.nn.relu(pre_activation, name=scope.name)
        colorization_level_conv3_upsampled = tf.image.resize_images(colorization_level_conv3, [112, 112], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        _activation_summary(colorization_level_conv3_upsampled)
                
    # colorization_level_conv4
    with tf.variable_scope('colorization_level_conv4') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 32], stddev=5e-2))
        conv = tf.nn.conv2d(colorization_level_conv3_upsampled, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[32]))
        pre_activation = tf.nn.bias_add(conv, biases)
        colorization_level_conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(colorization_level_conv4)
                
   # output_layer
    with tf.variable_scope('output_layer') as scope:
        weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 2], stddev=5e-2))
        conv = tf.nn.conv2d(colorization_level_conv4, weight, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[2]))
        pre_activation = tf.nn.bias_add(conv, biases)
        output_layer = tf.nn.sigmoid(pre_activation, name=scope.name)
        _activation_summary(output_layer)
        print("output_layer "  + str(output_layer.get_shape()))
    return output_layer

def trainNetwork():
    
  #with tf.Graph().as_default():
    #global_step = tf.train.get_or_create_global_step()
    path = os.path.dirname(os.path.realpath(__file__))
    file = "\\data\\tempData\\"
    dirname = path + file
    x_img_train, y_img_train = create(dirname)
    x_img_train_shape = np.shape(x_img_train)
    train_size = x_img_train_shape[0]
    print("train data complete")
    X_input_train = tf.placeholder(dtype = data_type(), shape = (FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1), name = "input_tensor")
    Y_output_train = tf.placeholder(dtype = data_type(), shape = None, name = "output_tensor")
    X_input_test = tf.placeholder(dtype = data_type(), shape = (FLAGS.batch_size_test, IMAGE_SIZE, IMAGE_SIZE, 1), name = "input_tensor")
    x_temp = np.reshape(x_img_train , (x_img_train_shape[0], x_img_train_shape[1], x_img_train_shape[2], x_img_train_shape[3]))
    print("x_temp shape " + str(x_temp.shape))
    logits = create_network(X_input_train)
    logits_norm = tf.image.convert_image_dtype(logits, tf.float32)
    logits_train = tf.image.resize_images(logits_norm, [224, 224], method=tf.image.ResizeMethod.BICUBIC)
    print("created")
    print("logits 2 shape " + str(logits_train.get_shape()))
#    color_output = tf.concat((x_temp, logits_train), axis = 3)
    #print("logits size " + str(logits.get_shape()))
    #logits_norm = cv2.normalize(logits, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=-1)
    #TypeError: src is not a numpy array, neither a scalar
    #logits_2 = cv2.resize(logits_norm, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_CUBIC)
   # x = cv2.resize(x_img[0], (224, 224, 1))
   # x = tf.reshape(x_img[0], [IMAGE_SIZE, IMAGE_SIZE, 1])
    #process test data
    filetest = "\\data\\test\\"
    testdirname = path + filetest
    x_img_test, y_img_test = create(testdirname)
    print("test data complete")
    #logits_test = create_network(X_input_test)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_output_train, logits=logits_train))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate = 0.001, rho = 0.95, epsilon = 1e-08).minimize(loss)
   # print(sess.run(loss, feed_dict = {X_input_train: x_img_train, Y_output_train: y_img_train}))
    #learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
    #sess.run(optimizer, feed_dict = {X_input_train: x_img_train, Y_output_train: y_img_train})    
    start_time = time.time()
    with tf.Session() as sess:
        print("session created")
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in xrange(int(NUM_EPOCHS * train_size) // FLAGS.batch_size):
          offset = (step * FLAGS.batch_size) % (train_size - FLAGS.batch_size)
          #type(x_img_train)
          batch_data = x_temp[offset:(offset + FLAGS.batch_size), :,:,:]
          batch_labels = y_img_train[offset:(offset + FLAGS.batch_size)]
          feed_dict = {X_input_train: batch_data, Y_output_train: batch_labels}
          if step % EVAL_FREQUENCY == 0:
            print("x temp batch shape " + str(batch_data.shape))
            color_output = tf.concat((batch_data, logits_train), axis = 3)
            l, predictions, color_imgs = sess.run([loss, logits_train, color_output], feed_dict=feed_dict)
            sess.run(optimizer, feed_dict=feed_dict)
            print("display images")
            display_images(color_imgs)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' %(step, float(step) * FLAGS.batch_size / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
            print('Minibatch loss: %.3f' % (l))
            print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
            
           # print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
           # print('Validation error: %.1f%%' % error_rate(
            #    eval_in_batches(validation_data, sess), validation_labels))
            sys.stdout.flush()
    # Finally print the result!
    #test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    ##print('Test error: %.1f%%' % test_error)
    #if FLAGS.self_test:
     # print('test_error', test_error)
     # assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (test_error,)

   # color_output = []
   #for i in range(FLAGS.batch_size):
   
   # print("optimizer " + str(optimizer))
    
    #logits_test = create_network(X_input_test)
    
    #correct_prediction = tf.equal(tf.argmax(logits), tf.argmax(Y_output))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

   # loss = loss(logits, labels)
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #train_op = build_model.train(loss, global_step)
    #sess.run(train_op, feed_dict = {})
    
if __name__ == '__main__':
    #sess = tf.InteractiveSession()
    trainNetwork()