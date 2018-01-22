import tensorflow as tf
import numpy
from functions import *
from parser import *
from PIL import Image
from visualizer import *
from net_read_and_run import *
from set_downsizer import *

net_syn = sys.argv[1]

net_num = sys.argv[2]

x = tf.placeholder(tf.float32, shape=[None, 3072], name='x_input')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
# x_im = tf.reshape(x, [-1, 32, 32, 1])

H = {'h0': tf.reshape(x, [-1, 32, 32, 3]),}

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
lR = tf.placeholder(tf.float32, shape=[])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#train_step = tf.train.MomentumOptimizer(learning_rate=lR, momentum=0.99).minimize(cross_entropy)

train_step = tf.train.AdamOptimizer(learning_rate=lR).minimize(cross_entropy)
#
#GradientDescentOptimizer(0.001)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

lR_val = 0.001
epochs = 100

inputs, labels = input_inject_CIFAR()
# inputs, labels = downsize_me(inputs_t, labels_t)
print len(inputs)
print len(labels)
# for i in xrange(epochs):
#     inputs.extend(inputs[:])
#     labels.extend(labels[:])
#print inputs[40000]
inputsTst, labelsTst = test_inject_CIFAR()
stopping = epochs
bsize=100
n_iters = (6000/bsize)
error_val = 100000
step_cnt = 0
val_data=[]
with tf.Session() as sess:

    # writer = tf.summary.FileWriter('output', graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())

    filname = '/home/hdft/Documents/DNN-Data-Run-tester/DATA_NETS_2017_' + str(net_num) + '.npy'

    data = numpy.array([[1,2,3,4]])

    print len(inputs)

    for j in xrange(epochs):
        print "Epoch: " + str(j)
        val_accuracy, val_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*60:bsize*62], y_:labels[bsize*60:bsize*62], lR:lR_val})
        print('Validation accuracy %g Error %f \n' % (val_accuracy, val_error))

        val_data.append([val_accuracy, val_error, j, 1])
        # if val_accuracy >= 0.98:
        #     stopping = j
        #     break

        if val_error > error_val:
            step_cnt += 1
        else:
            step_cnt = 0

        if error_val > val_error:
            error_val = val_error

        # error_val = val_error

        print step_cnt

        if step_cnt == 5:
            break




        for i in xrange(n_iters):
            #
            # test_accuracy, test_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[:2500], y_:labelsTst[:2500]})
            #
            # train_accuracy, error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*i:bsize*(i+1)], y_:labels[bsize*i:bsize*(i+1)]})
            #
            # data = numpy.concatenate((data, [[test_accuracy, test_error, train_accuracy, error]]), axis=0)
            if i%1 == 0:

                train_accuracy, error, learn = sess.run([accuracy, cross_entropy, lR], feed_dict={x:inputs[bsize*(i):bsize*(i+1)], y_:labels[bsize*(i):bsize*(i+1)], lR:lR_val})

                data = numpy.concatenate((data, [[train_accuracy, error, (n_iters*j)+i, 0]]), axis=0)

                #print('Step %d, Training accuracy %g Error %f Learning Rate %f \n' % (i, train_accuracy, error, learn))


            train_step.run(feed_dict={x:inputs[bsize*(i):bsize*(i+1)], y_:labels[bsize*(i):bsize*(i+1)], lR:lR_val})
        #lR_val *= 0.95




    print "\nTest\n"
    test_accuracy, test_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[:], y_:labelsTst[:], lR:lR_val})
    print "Accuracy = " + str(test_accuracy)
    print "Error = " + str(test_error)
    data = numpy.concatenate((data, [[test_accuracy, test_error, 0, 0]]), axis=0)
    data = numpy.concatenate((data, val_data), axis=0)

    numpy.save(filname, data)

    #writer.close()
