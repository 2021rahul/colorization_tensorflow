import tensorflow as tf

global_level_FC3 = tf.Variable(tf.ones(256))
mid_level_conv2 = tf.Variable(tf.zeros((28,28,256)))

mid_level_conv2_reshaped = tf.reshape(mid_level_conv2,[-1,256])
mid_level_conv2_reshaped = tf.unstack(mid_level_conv2_reshaped,axis=0)

fusion_level = [tf.concat([see_mid,global_level_FC3],0) for see_mid in mid_level_conv2_reshaped]
fusion_level = tf.stack(fusion_level)
fusion_level = tf.shape(fusion_level,[28,28,512])

with tf.Session() as create:
	tf.global_variables_initializer().run()
	print(mid_level_conv2.eval())
        print(global_level_FC3.eval())
	print(fusion_level.eval())
