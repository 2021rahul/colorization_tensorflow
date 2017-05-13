import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")

def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

biases_1 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
biases_2 = tf.Variable(tf.constant(0.0,shape=[64]))

with tf.Session() as session:
	tf.global_variables_initializer().run()
	print biases_1.eval()
	print biases_2.eval()