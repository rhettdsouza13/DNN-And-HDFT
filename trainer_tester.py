import tensorflow as tf
import numpy
from functions import *
from parser import *
from PIL import Image
from visualizer import *
from net_read_and_run import *
# from set_downsizer import *
import saliency
from matplotlib import pylab as P
import matplotlib
import tf_cnnvis

plot_dir = '/home/hdft/Documents/DNN-Complete/DNN-PLOTS/Box_Plots/'

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

def ShowImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  im = ((im + 1) * 127.5).astype(numpy.uint8)
  P.imshow(im)
  P.title(title)

numpy.set_printoptions(threshold=numpy.nan)
net_syn = sys.argv[1]
net_num = sys.argv[2]
run_num = int(sys.argv[3])
dim_size = int(sys.argv[4])

x = tf.placeholder(tf.float32, shape=[None, 12288], name='x_input')
y_ = tf.placeholder(tf.float32, shape=[None, 2], name='labels')
# x_im = tf.reshape(x, [-1, 32, 32, 1])
reshaper = tf.reshape(x, [-1, 3, 64, 64])
H = {'h0': tf.transpose(reshaper,[0,2,3,1]),}

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
neuron_selector = tf.placeholder(tf.int32)
y_neur = y[0][neuron_selector]


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#train_step = tf.train.MomentumOptimizer(learning_rate=lR, momentum=0.99).minimize(cross_entropy)

train_step = tf.train.AdamOptimizer(learning_rate=lR).minimize(cross_entropy)
# train_step = tf.train.MomentumOptimizer(learning_rate=lR, momentum=0.9).minimize(cross_entropy)

#GradientDescentOptimizer(0.001)
prediction = tf.argmax(y,1)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
casted = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(casted)



lR_val = 0.001
epochs = 100
with tf.device('/cpu:0'):
    inputs, labels = input_inject_MIT()
    # inputs, labels = downsize_me(inputs_t, labels_t)
    # inputs_total = numpy.load("/home/hdft/Documents/DNN-And-HDFT/downsized_CIFAR.npy")
    # numpy.random.shuffle(inputs_total)
    # inputs = [inp[0] for inp in inputs_total]
    # labels = [inp[1] for inp in inputs_total]
    # print len(inputs)
    # print len(labels)
    # print inputs[0]
    # print inputs[1]
    # for i in xrange(epochs):
    #     inputs.extend(inputs[:])
    #     labels.extend(labels[:])
    #print inputs[40000]
    # inputsTst, labelsTst = test_inject_CIFAR()
stopping = epochs
bsize=100
n_iters = (3300/bsize)
error_val = 100000
step_cnt = 0
val_data=[]
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('output', graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    # print inputs[0:1]
    # print sess.run([reshaper, H['h0']], feed_dict={x:inputs[0:1]})
    # print reshaper.get_shape()
    # filname = '/home/hdft/Documents/DNN-DataNEW-Run-224-param-MIT/DATA_NETS_2018_' + str(net_num) + '.npy'

    # filname = '/home/hdft/Documents/DNN-DataNEW-Run-' + str(run_num) + '-' + str(dim_size) + '-MIT/DATA_NETS_2018_' + str(net_num) + '.npy'

    data = numpy.array([[1,2,3,4]])

    print len(inputs)

    for j in xrange(epochs):
        print "Epoch: " + str(j)
        val_accuracy, val_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*25:bsize*33], y_:labels[bsize*25:bsize*33], lR:lR_val})
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

        if step_cnt == 50 or val_accuracy == 1 or j>100:
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

                # print('Step %d, Training accuracy %g Error %f Learning Rate %f \n' % (i, train_accuracy, error, learn))


            train_step.run(feed_dict={x:inputs[bsize*(i):bsize*(i+1)], y_:labels[bsize*(i):bsize*(i+1)], lR:lR_val})
        #lR_val *= 0.95



    # print "\nTest\n"
    # ta_1, te_1 = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[:2500], y_:labelsTst[:2500], lR:lR_val})
    # ta_2, te_2 = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[2500:5000], y_:labelsTst[2500:5000], lR:lR_val})
    # ta_3, te_3 = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[5000:7500], y_:labelsTst[5000:7500], lR:lR_val})
    # ta_4, te_4 = sess.run([accuracy, cross_entropy], feed_dict={x:inputsTst[7500:], y_:labelsTst[7500:], lR:lR_val})
    # test_accuracy = (ta_4+ta_3+ta_2+ta_1)*0.25
    # test_error = (te_4+te_3+te_2+te_1)*0.25
    # print "Accuracy = " + str(test_accuracy)
    # print "Error = " + str(test_error)
    # data = numpy.concatenate((data, [[test_accuracy, test_error, 0, 0]]), axis=0)
    # data = numpy.concatenate((data, val_data), axis=0)
    print "\nTest\n"
    test_accuracy, test_error = sess.run([accuracy, cross_entropy], feed_dict={x:inputs[bsize*33:], y_:labels[bsize*33:], lR:lR_val})
    print "Accuracy = " + str(test_accuracy)
    print "Error = " + str(test_error)
    data = numpy.concatenate((data, [[test_accuracy, test_error, 0, 0]]), axis=0)
    data = numpy.concatenate((data, val_data), axis=0)

    # numpy.save(filname, data)
    # print labels
    # matplotlib.rcParams.update({'font.size': 4.7})
    # ROWS = 4
    # COLS = 2
    # UPSCALE_FACTOR_COL = 4.5
    # UPSCALE_FACTOR_ROW = 1
    # P.figure(figsize=(ROWS * UPSCALE_FACTOR_ROW, COLS * UPSCALE_FACTOR_COL))
    #
    # prediction_class = sess.run(prediction, feed_dict={x:inputs[1:2]})[0]
    # print labels[1:2]
    # print prediction_class
    # #Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    # integrated_gradients = saliency.IntegratedGradients(tf.get_default_graph(), sess, y_neur, H['h0'])
    # im = sess.run(H['h0'], feed_dict={x:inputs[1:2]})[0]
    # # Baseline is a black image.
    # baseline = numpy.zeros(im.shape)
    # baseline.fill(-1)
    # # Compute the vanilla mask and the smoothed mask.
    # # vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
    # #     im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    # smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
    #     im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    #
    # # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    # # vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    # smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)
    # ShowImage(im,ax=P.subplot(ROWS, COLS, 1))
    # ShowGrayscaleImage(smoothgrad_mask_grayscale, title='Saliency Map', ax=P.subplot(ROWS, COLS, 2))
    # # Render the saliency masks.
    # # ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))
    #
    # prediction_class = sess.run(prediction, feed_dict={x:inputs[4:5]})[0]
    # print labels[4:5]
    # print prediction_class
    # #Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    # integrated_gradients = saliency.IntegratedGradients(tf.get_default_graph(), sess, y_neur, H['h0'])
    # im = sess.run(H['h0'], feed_dict={x:inputs[4:5]})[0]
    # # Baseline is a black image.
    # baseline = numpy.zeros(im.shape)
    # baseline.fill(-1)
    # # Compute the vanilla mask and the smoothed mask.
    # # vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
    # #     im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    # smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
    #     im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    #
    # # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    # # vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    # smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)
    # ShowImage(im,ax=P.subplot(ROWS, COLS, 3))
    # ShowGrayscaleImage(smoothgrad_mask_grayscale, title='Saliency Map', ax=P.subplot(ROWS, COLS, 4))
    #
    #
    # prediction_class = sess.run(prediction, feed_dict={x:inputs[5:6]})[0]
    # print labels[5:6]
    # print prediction_class
    # #Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    # integrated_gradients = saliency.IntegratedGradients(tf.get_default_graph(), sess, y_neur, H['h0'])
    # im = sess.run(H['h0'], feed_dict={x:inputs[5:6]})[0]
    # # Baseline is a black image.
    # baseline = numpy.zeros(im.shape)
    # baseline.fill(-1)
    # # Compute the vanilla mask and the smoothed mask.
    # # vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
    # #     im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    # smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
    #     im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    #
    # # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    # # vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    # smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)
    # ShowImage(im,ax=P.subplot(ROWS, COLS, 5))
    # ShowGrayscaleImage(smoothgrad_mask_grayscale, title='Saliency Map', ax=P.subplot(ROWS, COLS, 6))
    #
    # prediction_class = sess.run(prediction, feed_dict={x:inputs[2:3]})[0]
    # print labels[2:3]
    # print prediction_class
    # #Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    # integrated_gradients = saliency.IntegratedGradients(tf.get_default_graph(), sess, y_neur, H['h0'])
    # im = sess.run(H['h0'], feed_dict={x:inputs[2:3]})[0]
    # # Baseline is a black image.
    # baseline = numpy.zeros(im.shape)
    # baseline.fill(-1)
    # # Compute the vanilla mask and the smoothed mask.
    # # vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
    # #     im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    # smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
    #     im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    #
    # # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    # # vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    # smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)
    # ShowImage(im,ax=P.subplot(ROWS, COLS, 7))
    # ShowGrayscaleImage(smoothgrad_mask_grayscale, title='Saliency Map', ax=P.subplot(ROWS, COLS, 8))
    #
    # #save figure
    # P.savefig(plot_dir + 'Mit_saliency.png', dpi=300, format='png', bbox_inches='tight')
    # png2 = Image.open(plot_dir + 'Mit_saliency.png')
    # png2.save(plot_dir + 'Mit_saliency.tiff', compression='lzw')
    # tf_cnnvis.activation_visualization(sess, {x:inputs[1:2]}, input_tensor=None, layers=['c'], path_logdir='./Log', path_outdir='./Output')

    # P.show()

    for i,we in enumerate(Weights.values()):
        print we
        if i == 5:
            fils = sess.run(tf.transpose(we,(2,3,1,0)))
            # print fils
            # print sess.run(we)
            fig, axes = pl.subplots(3,32)
            fig.set_size_inches(11,2)
            matplotlib.rcParams.update({'font.size': 4.7})

            row=0
            col=0
            while row!=3 and col!=32:
                # print labels[i]
                im = fils[row][col]
                im = ((im + 1) * 127.5).astype(numpy.uint8)
                im = Image.fromarray(im, mode='L')
                im = im.resize((25,25))
                im = numpy.asarray(im)
                axes[row,col].imshow(im, cmap='gray')
                # axes[0].set_title('Image')
                axes[row,col].get_xaxis().set_visible(False)
                axes[row,col].get_yaxis().set_visible(False)
                # print row, col
                if col == 31:
                    row += 1
                    col=0
                else:
                    col += 1
            base_path = '/home/hdft/Documents/DNN-Complete/DNN-PLOTS/Box_Plots/'

            fig.savefig(base_path + 'filters.png', dpi=300, format='png', bbox_inches='tight')
            png2 = Image.open(base_path + 'filters.png')
            png2.save(base_path + 'filters.tiff', compression='lzw')
            pl.show()
