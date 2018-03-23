import matplotlib.pyplot as pl
import os
import numpy
from sklearn.kernel_ridge import KernelRidge
from fitter import *

data_dir = '/home/hdft/Documents/DNN-Data/'
data_dir_r2 = '/home/hdft/Documents/DNN-Data-Run-205-param-CIFAR-4000-1000/'
archive5000 = '/home/hdft/Documents/DNN-Complete/1111-PATH-Run/'

run=3

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

dirs = os.listdir(data_dir_r2)
fils_list = []
# nets_file = open("nets_list6.txt",'r')
# nets_list = netfile.readlines()
com_file = open("common.txt", 'r')
common_list = com_file.readlines()
print common_list
netfile_7 = open("param_list1000run.txt", 'r')
nets_7 = netfile_7.readlines()
good_nets = open("good_nets_ol.txt", "w+")
if run == 2:
    train_acc_plot = []
    train_err_plot = []
    val_acc_plot = []
    val_err_plot = []
    of_plot = []
    of_plot_error = []
    file_list = os.listdir(data_dir_r2)
    of_flag = 0
    for d_file in os.listdir(data_dir_r2):
        print d_file
        data = numpy.load(data_dir_r2 + d_file)
import matplotlib.pyplot as pl
import os
import numpy
from sklearn.kernel_ridge import KernelRidge
from fitter import *

data_dir = '/home/hdft/Documents/DNN-Data/'
data_dir_r2 = '/home/hdft/Documents/DNN-Data-Run-205-param-CIFAR-4000-1000/'

run=3

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

dirs = os.listdir(data_dir_r2)
fils_list = []
# nets_file = open("nets_list6.txt",'r')
# nets_list = netfile.readlines()
com_file = open("common.txt", 'r')
common_list = com_file.readlines()
print common_list
netfile_7 = open("param_list1000run.txt", 'r')
nets_7 = netfile_7.readlines()
good_nets = open("good_nets_ol.txt", "w+")
if run == 2:
    train_acc_plot = []
    train_err_plot = []
    val_acc_plot = []
    val_err_plot = []
    of_plot = []
    of_plot_error = []
    file_list = os.listdir(data_dir_r2)
    of_flag = 0
    for d_file in os.listdir(data_dir_r2):
        print d_file
        data = numpy.load(data_dir_r2 + d_file)

        train_accuracy = []
        train_error = []
        val_accuracy = []
        val_error = []
        test_accuracy = []
        test_error = []
        iterat = 0
        if 1:
            for tup in data:
                if tup[3] == 0:
                    train_accuracy.append(tup[0])
                    train_error.append(tup[1])
                if tup[3] == 1:
                    val_accuracy.append(tup[0])
                    val_error.append(tup[1])
                    if tup[2]>=0 and tup[1] <= 0.22:
                        # print tup[2]
                        # print tup[0]
                        # print d_file
                        of_flag = 1
            iterat+=1

        if of_flag == 1:
            # print tup[2]
            # print tup[0]
            print d_file
            print len(val_accuracy)
            of_plot.append(val_accuracy)

def apply_regression(y):
    clf = KernelRidge(alpha=10.0, kernel='rbf', degree=3, gamma=0.1)
    iters = numpy.reshape(range(len(y)), (-1,1))
    ys = numpy.reshape(y, (-1,1))
    clf.fit(iters, ys)
    return clf.predict(iters)

if run == 3:
    train_acc_plot = []
    train_err_plot = []
    val_acc_plot = []
    val_err_plot = []
    of_plot = []
    of_plot_error = []
    file_list = os.listdir(data_dir_r2)
    of_flag = 0
    for run in os.listdir(archive5000):
        print run
        for d_file in os.listdir(archive5000 + '/' + run):
            # print d_file
            data = numpy.load(archive5000 + '/' + run + '/' + d_file)

            train_accuracy = []
            train_error = []
            val_accuracy = []
            val_error = []
            test_accuracy = []
            test_error = []
            iterat = 0
            if 1:
                for tup in data:
                    if tup[3] == 0:
                        train_accuracy.append(tup[0])
                        train_error.append(tup[1])
                    if tup[3] == 1:
                        val_accuracy.append(tup[0])
                        val_error.append(tup[1])
                        if tup[2]>=0 and tup[1] <= 0.22:
                            # print tup[2]
                            # print tup[0]
                            # print d_file
                            of_flag = 1
                iterat+=1

            if of_flag == 1:
                # print tup[2]
                # print tup[0]
                print d_file
                print len(val_accuracy)
                of_plot.append(val_accuracy)
                of_plot_error.append(val_error)
                net_name = nets_7[int(d_file.split('_')[3].split('.')[0])]
                print net_name,

                print str(max(val_accuracy))
                print str(min(val_error))
                print train_accuracy[-1]
                of_flag = 0

            train_acc_plot.append(train_accuracy)
            train_err_plot.append(train_error)
            val_acc_plot.append(val_accuracy)
            val_err_plot.append(val_error)
            fils_list.append(d_file)


    pl.figure(1)
    pl.xlabel("Epoch Number")
    pl.ylabel("Validation Error")
    for plots in val_err_plot:
        line, = pl.plot(plots[1:])

    # pl.figure(2)
    # for plots in train_err_plot:
    #     line, = pl.plot(plots)


    pl.figure(3)
    pl.xlabel("Epoch Number")
    pl.ylabel("Validation Accuracy")
    for plots in val_acc_plot:

        line, = pl.plot(range(len(plots)-1), plots[1:])
    # pl.figure(4)
    # for plots in train_acc_plot:
    #     line, = pl.plot(range(len(plots)), plots)

    test_scatter = []
    for plots in train_acc_plot:
        if len(plots) >= 1:
            test_scatter.append(plots[len(plots)-1])


    pl.figure(8)
    pl.scatter(range(len(test_scatter)), test_scatter)
    # for i in xrange(len(test_scatter)):
        # if test_scatter[i] > 0.75:
        #     print i
    # plot100 = [(i-0.95)*100 for i in val_acc_plot[0][1:]]

    pl_ind = 0
    pl.figure(5)
    for plots in val_acc_plot:
        try:
            opt, cov = fitter([float(i) for i in range(len(plots[1:]))], plots[1:])
        except:
            # print "Issue, Moving On"
            # print pl_ind
            continue


        #pl.plot(plots[1:])
        val_fit=[func(i,opt[0],opt[1],opt[2]) for i in range(len(plots[1:]))]

        # pl.plot(func(val_acc_plot[0][1:], *opt), 'r-', label="Fitted Curve")
        pl.plot(val_fit)
        #print pl_ind
        pl_ind+=1
        # print opt

    pl.figure(6)
    pl.xlabel("Epoch Number")
    pl.ylabel("Validation Accuracy")
    for plots in of_plot:
        line, = pl.plot(range(len(plots)-1), plots[1:])
    pl.figure(7)
    pl.xlabel("Epoch Number")
    pl.ylabel("Validation Error")
    for plots in of_plot_error:
        line, = pl.plot(range(len(plots)-1), plots[1:])



    pl.show()
