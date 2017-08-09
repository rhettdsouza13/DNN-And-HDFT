import tensorflow as tf
import numpy
from functions import *
from parser import *
from PIL import Image
from visualizer import *
from net_read_and_run import *

net_syn = sys.argv[1]

net_num = sys.argv[2]

x = tf.placeholder(tf.float32, shape=[None, 1024], name='x_input')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
# x_im = tf.reshape(x, [-1, 32, 32, 1])

H = {'h0': tf.reshape(x, [-1, 32, 32, 1]),}

#H['h0'] = max_pool_kxk(x_im, 2)




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

            if ind==(len(fn_list)-1):

                H[h_key] = tf.matmul(H[h_key_resh],Weights[w_key])+Bias[b_key]
                print "Here"
            else:
                H[h_key] = tf.nn.relu(tf.matmul(H[h_key_resh],Weights[w_key])+Bias[b_key])

            fc_flag=1

        elif ind==(len(fn_list)-1):
            print "Here"
            H[h_key] = tf.matmul(H[pre_h_key],Weights[w_key])+Bias[b_key]

        else:
            H[h_key] = tf.nn.relu(tf.matmul(H[pre_h_key],Weights[w_key])+Bias[b_key])


    if fn_list[ind][0] == 'conv':
        H[h_key] = tf.nn.relu(conv2d(H[pre_h_key],Weights[w_key])+Bias[b_key])

    if fn_list[ind][0] == 'max_pooling':
        H[h_key] = max_pool_kxk(H[pre_h_key], fn_list[ind][2])
print H
print Weights
print Bias

h_out_key = 'h' + str(len(fn_list))
y = H[h_out_key]

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


epochs = 5
inputs, labels = input_inject()
for i in xrange(epochs):
    inputs.extend(inputs[:])
    labels.extend(labels[:])
inputsTst, labelsTst = test_inject()
bsize=100
n_iters = (60000/bsize)*epochs
with tf.Session() as sess:

    # writer = tf.summary.FileWriter('output', graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())

    filname = '/home/hdft/Documents/DNN-Data/DATA_NETS_2017_' + str(net_num) + '.npy'

    data = numpy.array([[1,2,3,4,5]])

    upperlim = 5


    for i in xrange(n_iters):
        #
        # test_accuracy, test_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[:2500], y_:labelsTst[:2500]})
        #
        # train_accuracy, error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*i:bsize*(i+1)], y_:labels[bsize*i:bsize*(i+1)]})
        #
        # data = numpy.concatenate((data, [[test_accuracy, test_error, train_accuracy, error]]), axis=0)
        if i == upperlim or i == n_iters :
            #print inputs[30*i], labels[30*i]
            test_accuracy, test_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[:], y_:labelsTst[:]})

            train_accuracy, error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*(i-5):bsize*(i)], y_:labels[bsize*(i-5):bsize*(i)]})

            data = numpy.concatenate((data, [[test_accuracy, test_error, train_accuracy, error, i]]), axis=0)

            print "Step "+ str(i) + " Testing accuracy = " + str(test_accuracy) + " Error " + str(test_error)
            print('Step %d, Training accuracy %g Error %f \n' % (i, train_accuracy, error))

            upperlim *= 2



        train_step.run(feed_dict={x:inputs[bsize*(i):bsize*((i+1))], y_:labels[bsize*(i):bsize*((i+1))]})

    numpy.save(filname, data)


    #
    # print "\nTest\n"
    # test_accuracy = sess.run(accuracy, feed_dict={x:inputsTst[:], y_:labelsTst[:]})
    # print "accuracy = " + str(test_accuracy)

    #writer.close()
