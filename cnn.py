import tensorflow as tf
import numpy
from functions import *
from parser import *
from PIL import Image
from visualizer import *

x = tf.placeholder(tf.float32, shape=[None, 1024])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W1 = weight_initializer([5,5,1,32])
b1 = bias_initializer([32])

x_re = tf.reshape(x, [-1,32,32,1])

h_conv1 = tf.nn.relu(conv2d(x_re, W1) + b1)
h_pool1 = max_pool_kxk(h_conv1,2)

W2 = weight_initializer([5,5,32,64])
b2 = bias_initializer([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool_kxk(h_conv2,2)

W3 = weight_initializer([8*8*64, 1024])
b3 = bias_initializer([1024])

h_pool2_flatten = tf.reshape(h_pool2, [-1,8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W3)+b3)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

Wout = weight_initializer([1024,10])
bout = bias_initializer([10])

y_conv = tf.matmul(h_fc1_drop, Wout) + bout
# y_conv = tf.matmul(x, Wout) + bout


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

inputs, labels = input_inject()
inputsTst, labelsTst = test_inject()
bsize=100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    view32x32Im(h_conv1.eval(feed_dict={x:[inputsTst[0]], y_:[labelsTst[0]], keep_prob:1.0}))

    for i in xrange(600):
        if i%10==0:
            #print inputs[30*i], labels[30*i]

            test_accuracy, test_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[:1000], y_:labelsTst[:1000], keep_prob:1.0})
            print "Step "+ str(i) + " Testing accuracy = " + str(test_accuracy) + " Error " + str(test_error)
            train_accuracy, error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*i:bsize*(i+1)],
            y_:labels[bsize*i:bsize*(i+1)], keep_prob:1.0})
            print('step %d, training accuracy %g error %f \n' % (i, train_accuracy, error))
        train_step.run(feed_dict={x:inputs[bsize*i:bsize*(i+1)], y_:labels[bsize*i:bsize*(i+1)], keep_prob:0.5})



    print "\nTest\n"
    test_accuracy = sess.run(accuracy, feed_dict={x:inputsTst[:1000], y_:labelsTst[:1000], keep_prob:1.0})
    print "accuracy = " + str(test_accuracy)

    weight= W2.eval()

    view32x64(weight)


    # print labelsTst[0]
    # print conv1st
    #print weight
    view32x32Im(x_re.eval(feed_dict={x:[inputsTst[0]], y_:[labelsTst[0]], keep_prob:1.0}))
    # conv2nd = h_conv2.eval(feed_dict={x:[inputsTst[0]], y_:[labelsTst[0]], keep_prob:1.0})
    # view16x16Im(conv2nd)
    # view1x32(weight)
