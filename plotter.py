import matplotlib.pyplot as pl
import os
import numpy
from sklearn.kernel_ridge import KernelRidge


data_dir = '/home/hdft/Documents/DNN-Data/'

def apply_regression(y):
    clf = KernelRidge(alpha=0.075, kernel='rbf', degree=3, gamma=0.1)
    iters = numpy.reshape(range(len(y)), (-1,1))
    ys = numpy.reshape(y, (-1,1))
    clf.fit(iters, ys)
    return clf.predict(iters)


#legendHandles = []
counter=0
bad_err=[]
bad=[]
acc_plot=[]
pl.figure(1)
for data_file in os.listdir(data_dir):

    data = numpy.load(data_dir+data_file)
    test_accuracy_r= data[1:,0]
    test_error_r= data[1:,1]
    train_accuracy_r= data[1:,2]
    train_error_r = data[1:,3]

    test_accuracy = apply_regression(test_accuracy_r)

    acc_plot.append(test_accuracy)

    test_error = apply_regression(test_error_r)
    #train_error = apply_regression(train_error_r)

    line, = pl.plot(range(len(test_error)),test_error, label="Test Error" + str(counter))
    #legendHandles.append(line)
    if test_error[90] > 0.25:
        bad_err.append(data_file)
        if test_accuracy[90] < 0.898:
            print data_file
            bad.append(data_file)

    # line, = pl.plot(range(len(train_error_r)),train_error_r, label="Train Error" + str(counter))
    # legendHandles.append(line)

    counter+=1

pl.figure(2)
for plots in acc_plot:
    line, = pl.plot(range(len(plots)),plots, label="Test accuracy" + str(counter))
    counter+=1

# pl.legend(handles = legendHandles)
with open('nets_list.txt', 'r') as netfile:
    nets = netfile.readlines()
    for bad_n in bad:
        nam_syn = bad_n.split('_')
        bad_ind = int(nam_syn[3].split('.')[0])
        print nets[bad_ind]

pl.show()
