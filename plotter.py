import matplotlib.pyplot as pl
import os
import numpy
from sklearn.kernel_ridge import KernelRidge


data_dir = '/home/hdft/Documents/DNN-Data/'
data_dir_r2 = '/home/hdft/Documents/DNN-Data-Run-2/'

run=2

def apply_regression(y):
    clf = KernelRidge(alpha=10.0, kernel='rbf', degree=3, gamma=0.1)
    iters = numpy.reshape(range(len(y)), (-1,1))
    ys = numpy.reshape(y, (-1,1))
    clf.fit(iters, ys)
    return clf.predict(iters)

if run == 1:
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

    #DATA_NETS_2017_3302.npy check network????


if run == 2:
    train_acc_plot = []
    train_err_plot = []
    val_acc_plot = []
    val_err_plot = []
    for d_file in os.listdir(data_dir_r2):
        data = numpy.load(data_dir_r2 + d_file)

        train_accuracy = []
        train_error = []
        val_accuracy = []
        val_error = []
        test_accuracy = []
        test_error = []
        for tup in data:
            if tup[3] == 0:
                train_accuracy.append(tup[0])
                train_error.append(tup[1])
            if tup[3] == 1:
                val_accuracy.append(tup[0])
                val_error.append(tup[1])
        train_acc_plot.append(train_accuracy)
        train_err_plot.append(train_error)
        val_acc_plot.append(val_accuracy)
        val_err_plot.append(val_error)

    pl.figure(1)
    for plots in val_err_plot:
        line, = pl.plot(plots)

    pl.figure(2)
    for plots in train_err_plot:
        line, = pl.plot(plots)


    pl.figure(3)
    for plots in val_acc_plot:
        line, = pl.plot(range(len(plots)), plots)

    pl.figure(4)
    for plots in train_acc_plot:
        line, = pl.plot(range(len(plots)), plots)

    pl.show()
