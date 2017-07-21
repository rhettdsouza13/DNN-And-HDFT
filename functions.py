import tensorflow as tf

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_kxk(x,k):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding="SAME")

def weight_initializer(shape):
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W)

def bias_initializer(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
