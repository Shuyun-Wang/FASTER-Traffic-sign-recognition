import numpy as np
import tensorflow as tf 
 
slim = tf.contrib.slim

def res_18_invariant(input_tensor, conv_depth, kernel_shape, layer_name):
	# gamma_init = tf.random_normal_initializer(1., 0.02)
	with tf.variable_scope(layer_name):
		relu = tf.nn.relu(slim.conv2d(input_tensor, conv_depth, kernel_shape))
		outputs = tf.nn.relu(slim.conv2d(relu, conv_depth, kernel_shape) + input_tensor)
	return outputs
 
def res_18_change(input_tensor, conv_depth, kernel_shape, layer_name):
	with tf.variable_scope(layer_name):
		relu = tf.nn.relu(slim.conv2d(input_tensor, conv_depth, kernel_shape, stride=2))
		input_tensor_reshape = slim.conv2d(input_tensor, conv_depth, [1,1], stride=2)
	outputs = tf.nn.relu(slim.conv2d(relu, conv_depth, kernel_shape) + input_tensor_reshape)
	return outputs


def res_50_invariant(input_tensor, conv_depth, kernel_shape, layer_name):
    with tf.variable_scope(layer_name):
        relu = tf.nn.relu(slim.conv2d(input_tensor, conv_depth/2,[1,1]))
        relu_next = tf.nn.relu(slim.conv2d(relu, conv_depth/2, kernel_shape))
        input_tensor_reshape = slim.conv2d(relu_next, conv_depth, [1,1])
    outputs = tf.nn.relu(slim.conv2d(input_tensor, conv_depth, [1,1]) + input_tensor_reshape)
    return outputs


def res_50_change(input_tensor, conv_depth, kernel_shape, layer_name):
    with tf.variable_scope(layer_name):
        relu = tf.nn.relu(slim.conv2d(input_tensor, conv_depth/2,[1,1]))
        relu_next = tf.nn.relu(slim.conv2d(relu, conv_depth/2, kernel_shape, stride=2))
        input_tensor_reshape = slim.conv2d(relu_next, conv_depth*2, [1,1])
    outputs = tf.nn.relu(slim.conv2d(input_tensor, conv_depth*2, [1,1],stride=2) + input_tensor_reshape)
    return outputs



def model_specification(inputs):
	x = tf.reshape(inputs, [-1, 48, 48, 1])
	conv_1 = tf.nn.relu(slim.conv2d(x, 32, [5, 5])) #48*48*32
	pool_1 = slim.max_pool2d(conv_1, [3, 3]) #24*24*32

	block_1 = res_18_invariant(pool_1, 32, [3, 3], 'layer_2')
	block_2 = res_18_invariant(block_1, 32, [3, 3], 'layer_3')

	block_3 = res_18_invariant(block_2, 32, [3, 3], 'layer_4') #24*24*32
	block_4 = res_18_change(block_3, 64, [3, 3], 'layer_5')

	block_5 = res_18_invariant(block_4, 64, [3, 3], 'layer_6')
	block_6 = res_18_invariant(block_5, 64, [3, 3], 'layer_7')

	block_7 = res_50_change(block_6, 64, [3, 3], 'layer_8')
	block_8 = res_50_invariant(block_7, 128, [3, 3], 'layer_9')

	net_flatten = slim.flatten(block_8, scope='flatten')
	fc_1 = slim.fully_connected(slim.dropout(net_flatten, 0.8), 1000, activation_fn=tf.nn.tanh, scope='fc_1')
	output = slim.fully_connected(slim.dropout(fc_1, 0.8), 62, activation_fn=None, scope='output_layer')
	return output
