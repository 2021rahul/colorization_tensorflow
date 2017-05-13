import tensorflow as tf
import numpy as np
mid_level_conv2 = tf.Variable(np.zeros((28,28,256)))
global_level_FC3 = tf.Variable(np.zeros(256))
reshape = tf.reshape(mid_level_conv2,[-1,256])

'''
mat_global_level_FC3 = tf.matmul(tf.Variable(np.ones()))
'''

# augment = tf.Variable(np.ones((reshape.get_shape()[0],1)))
# mat_b = tf.matmul(augment,b)
# fused = tf.pack([reshape,mat_b],axis=1)
# fused_reshape = tf.reshape(fused, [9,-1])
# fused_reshape = tf.reshape(fused,[3,3,6])
# weights = tf.Variable(np.ones((3,6)))
# bias = tf.Variable(fused_reshape[0,0,:])
# fusion_xy = tf.matmul(weights,fused_el)

with tf.Session() as session:
	tf.initialize_all_variables().run()
	print (mid_level_conv2.eval())
	print (global_level_FC3.eval())
	print (reshape.eval())
	# print (augment.eval())
	# print (mat_b.eval())
	# print (fused.eval())
	# print (fused_reshape.eval())
	# print (weights.eval())
	# print (bias.eval())
	# print (fused_el.eval())
	# print (fusion_xy.eval())
