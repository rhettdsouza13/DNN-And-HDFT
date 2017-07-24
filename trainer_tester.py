import tensorflow as tf
import numpy
from functions import *
from parser import *
from PIL import Image
from visualizer import *
from net_read_and_run import *

net_syn = sys.argv[1]


x = tf.placeholder(tf.float32, shape=[None, 256])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

H = {'h0' : tf.reshape(x, [-1, 16, 16, 1])}


Weights, Bias, fn_list, ch_list = set_variables(net_syn)

fc_flag = 0

for ind in xrange(len(fn_list)):
    pre_h_key =  'h' + str(ind)
    h_key = 'h' + str(ind+1)
    w_key = 'W' + str(ind)
    b_key = 'b' + str(ind)

    if fn_list[ind][0] == 'full':
        if fc_flag==0:
            h_key_resh = 'h' + str(ind+1) + '_resh'
            H[h_key_resh] = tf.reshape(H[pre_h_key], [-1, Weights[w_key].get_shape()[0].value])

            H[h_key] = tf.nn.relu(tf.matmul(H[h_key_resh],Weights[w_key])+Bias[b_key])

        else:
            H[h_key] = tf.nn.relu(tf.matmul(H[h_key],Weights[w_key])+Bias[b_key])
            fc_flag=1

    if fn_list[ind][0] == 'conv':
        H[h_key] = tf.nn.relu(conv2d(H[pre_h_key],Weights[w_key])+Bias[b_key])

    if fn_list[ind][0] == 'max_pooling':
        H[h_key] = max_pool_kxk(H[pre_h_key], fn_list[ind][2])
print H


h_out_key = 'h' + str(len(fn_list)-1)

y = H[h_out_key]

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

inputs, labels = input_inject()
inputsTst, labelsTst = test_inject()
bsize=100
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())


    for i in xrange(600):
        if i%10==0:
            #print inputs[30*i], labels[30*i]

            test_accuracy, test_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[:1000], y_:labelsTst[:1000]})
            print "Step "+ str(i) + " Testing accuracy = " + str(test_accuracy) + " Error " + str(test_error)
            train_accuracy, error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*i:bsize*(i+1)],
            y_:labels[bsize*i:bsize*(i+1)], keep_prob:1.0})
            print('step %d, training accuracy %g error %f \n' % (i, train_accuracy, error))
        train_step.run(feed_dict={x:inputs[bsize*i:bsize*(i+1)], y_:labels[bsize*i:bsize*(i+1)]})



    print "\nTest\n"
    test_accuracy = sess.run(accuracy, feed_dict={x:inputsTst[:1000], y_:labelsTst[:1000]})
    print "accuracy = " + str(test_accuracy)
