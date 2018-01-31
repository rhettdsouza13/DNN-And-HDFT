import tensorflow as tf
import numpy

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_kxk(x,k):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding="SAME")

def weight_initializer(shape):
    # if len(shape) == 4:
    #     X = numpy.random.random((shape[3], shape[0]*shape[1]*shape[2]))
    # if len(shape) == 2:
    #     X = numpy.random.random((shape[1], shape[0]))
    # U, _, Vt = numpy.linalg.svd(X, full_matrices=False)
    # print Vt.shape
    # print numpy.allclose(numpy.dot(Vt, Vt.T), numpy.eye(Vt.shape[0]))
    # if len(shape) == 4:
    #     W = Vt.reshape((shape[0], shape[1], shape[2], shape[3]))
    # if len(shape) == 2:
    #     W = Vt.reshape((shape[0], shape[1]))
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W)

def bias_initializer(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# print weight_initializer([1024,601])
