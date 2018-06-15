from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 224
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_img(filename):
	img = cv2.imread(filename,3)
	height, width, channels = img.shape
	greyimg = cv2.cvtColor(cv2.resize(img, (224, 224)),cv2.COLOR_BGR2GRAY)
	colorimg = cv2.cvtColor(cv2.resize(img, (224, 224)),cv2.COLOR_BGR2LAB)
	return greyimg, colorimg

def create_input(dirname):
	filenames = os.listdir(dirname)
	x_img = []
	y_img = []
	for filename in filenames:
		grey, color = read_img(dirname+filename)
		x_img.append(grey)
		y_img.append(color)
	return x_img,y_img

def read_image(filename):
	
	class ImageRecord(object):
		pass

	result = ImageRecord()
	img = cv2.imread(filename,3)
	height, width, channels = img.shape
	greyimg = cv2.cvtColor(cv2.resize(img, (224, 224)),cv2.COLOR_BGR2GRAY)
	colorimg = cv2.cvtColor(cv2.resize(img, (224, 224)),cv2.COLOR_BGR2LAB)

	result.label = tf.constant(colorimg)
	result.image = tf.constant(greyimg)

	return result

filename = "./data/train/09-322-02-1.bmp"
with tf.Session() as session:
	tf.global_variables_initializer().run()
	result = read_image(filename)
	print(result.image)
	print(result.label)
	cv2.imshow(result.image)
	cv2.waitKey(0)