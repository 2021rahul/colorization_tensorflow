import tensorflow as tf
import numpy as np
mid_level_conv2 = tf.Variable(np.zeros((28,28,256)))
global_level_FC3 = tf.Variable(np.ones(256))
mid_level_conv2_np = tf.reshape(mid_level_conv2,[-1,256])
fusion_layer_np = [x+global_level_FC3_np for x in mid_level_conv2_np]
# fusion_layer_np = np.asarray(fusion_layer_np)
# fusion_layer_np = np.reshape(fusion_layer_np,mid_level_conv2.get_shape())



with tf.Session() as create:
	tf.global_variables_initializer().run()
	mid_level_conv2_np = mid_level_conv2.eval()
	global_level_FC3_np = global_level_FC3.eval()
	mid_level_conv2_np = np.reshape(mid_level_conv2_np,[-1,256])
	# global_level_FC3_np = np.reshape(global_level_FC3_np,[-1,256])
	fusion_layer_np = [x+global_level_FC3_np for x in mid_level_conv2_np]
	fusion_layer_np = np.asarray(fusion_layer_np)
	fusion_layer_np = np.reshape(fusion_layer_np,mid_level_conv2.get_shape())

fusion_layer = tf.Variable(fusion_layer_np)
print (fusion_layer.get_shape())
	# print (fusion_layer_np)
	# print (fusion_layer_np.shape)
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

# with tf.Session() as session:
# 	tf.initialize_all_variables().run()
# 	print (mid_level_conv2.eval())
# 	print (global_level_FC3.eval())
# 	print (reshape.eval())
# 	print (mid_level_conv2.get_shape())
# 	print (global_level_FC3.get_shape())
# 	print (reshape.get_shape())
	# print (augment.eval())
	# print (mat_b.eval())
	# print (fused.eval())
	# print (fused_reshape.eval())
	# print (weights.eval())
	# print (bias.eval())
	# print (fused_el.eval())
	# print (fusion_xy.eval())
