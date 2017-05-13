from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import input_create

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")

IMAGE_SIZE = [224,224]
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _activation_summary(x):
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))

def create_network():

	image = tf.placeholder("float", [None, 224, 224, 1])

	# low_level_conv1
	with tf.variable_scope('low_level_conv1') as scope:
		weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 64],stddev=5e-2))
	    bias = tf.Variable(tf.constant(0.0,shape=[64]))
	    conv = tf.nn.conv2d(image, weight, [1, 2, 2, 1], padding="SAME")
	    low_level_conv1 = tf.nn.relu(tf.nn.bias_add(conv, bias))
	    _activation_summary(low_level_conv1)

	# low_level_conv2
	with tf.variable_scope('low_level_conv1') as scope:
		weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128],stddev=5e-2))
	    bias = tf.Variable(tf.constant(0.0,shape=[128]))
	    conv = tf.nn.conv2d(low_level_conv1, weight, [1, 1, 1, 1], padding="SAME")
	    low_level_conv2 = tf.nn.relu(tf.nn.bias_add(conv, bias))
	    _activation_summary(low_level_conv2)

	# low_level_conv2
	with tf.variable_scope('low_level_conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 64, 128],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(low_level_conv1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		low_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(low_level_conv2)

	# low_level_conv3
	with tf.variable_scope('low_level_conv3') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 128, 128],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(low_level_conv2, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		low_level_conv3 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(low_level_conv3)

	# low_level_conv4
	with tf.variable_scope('low_level_conv4') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 128, 256],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(low_level_conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		low_level_conv4 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(low_level_conv4)    

	# low_level_conv5
	with tf.variable_scope('low_level_conv5') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 256, 256],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(low_level_conv4, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		low_level_conv5 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(low_level_conv5)

	# low_level_conv6
	with tf.variable_scope('low_level_conv6') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 256, 512],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(low_level_conv5, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		low_level_conv6 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(low_level_conv6)

	# mid_level_conv1
	with tf.variable_scope('mid_level_conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 512, 512],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(low_level_conv6, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		mid_level_conv1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(mid_level_conv1)

	# mid_level_conv2
	with tf.variable_scope('mid_level_conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 512, 256],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(mid_level_conv1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		mid_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(mid_level_conv2)

	# global_level_conv1
	with tf.variable_scope('global_level_conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 512, 512],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(low_level_conv6, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		global_level_conv1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(global_level_conv1)

	# global_level_conv2
	with tf.variable_scope('global_level_conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 512, 512],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(global_level_conv1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		global_level_conv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(global_level_conv2)

	# global_level_conv3
	with tf.variable_scope('global_level_conv3') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 512, 512],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(global_level_conv2, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		global_level_conv3 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(global_level_conv3)

	# global_level_conv4
	with tf.variable_scope('global_level_conv4') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[3, 3, 512, 512],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(global_level_conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		global_level_conv4 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(global_level_conv4)

	# global_level_FC1
	with tf.variable_scope('global_level_FC1') as scope:
		reshape = tf.reshape(global_level_conv4, [FLAGS.batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights',
											  shape=[dim, 1024],
											  stddev=0.04,
											  wd=0.004)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
		global_level_FC1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
		_activation_summary(global_level_FC1)
	
	# global_level_FC2
	with tf.variable_scope('global_level_FC2') as scope:
		weights = _variable_with_weight_decay('weights',
											  shape=[1024, 512],
											  stddev=0.04,
											  wd=0.004)
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
		global_level_FC2 = tf.nn.relu(tf.matmul(global_level_FC1, weights) + biases, name=scope.name)
		_activation_summary(global_level_FC2)

	# global_level_FC3
	with tf.variable_scope('global_level_FC3') as scope:
		weights = _variable_with_weight_decay('weights',
											  shape=[512, 256],
											  stddev=0.04,
											  wd=0.004)
		biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
		global_level_FC3 = tf.nn.relu(tf.matmul(global_level_FC2, weights) + biases, name=scope.name)
		_activation_summary(global_level_FC3)

	#	

  return softmax_linear

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
	  logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
	total_loss: Total loss from loss().
  Returns:
	loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
	# Name each loss as '(raw)' and name the moving average version of the loss
	# as the original loss name.
	tf.contrib.deprecated.scalar_summary(l.op.name + ' (raw)', l)
	tf.contrib.deprecated.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Args:
	total_loss: Total loss from loss().
	global_step: Integer Variable counting the number of training steps
	  processed.
  Returns:
	train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
								  global_step,
								  decay_steps,
								  LEARNING_RATE_DECAY_FACTOR,
								  staircase=True)
  tf.contrib.deprecated.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
	opt = tf.train.GradientDescentOptimizer(lr)
	grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
	tf.contrib.deprecated.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
	if grad is not None:
	  tf.contrib.deprecated.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
	  MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
	train_op = tf.no_op(name='train')

  return train_op


# def maybe_download_and_extract():
#   """Download and extract the tarball from Alex's website."""
#   dest_directory = FLAGS.data_dir
#   if not os.path.exists(dest_directory):
# 	os.makedirs(dest_directory)
#   filename = DATA_URL.split('/')[-1]
#   filepath = os.path.join(dest_directory, filename)
#   if not os.path.exists(filepath):
# 	def _progress(count, block_size, total_size):
# 	  sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
# 		  float(count * block_size) / float(total_size) * 100.0))
# 	  sys.stdout.flush()
# 	filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
# 	print()
# 	statinfo = os.stat(filepath)
# 	print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

#   tarfile.open(filepath, 'r:gz').extractall(dest_directory)
