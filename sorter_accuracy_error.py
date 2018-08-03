import os
import numpy
import matplotlib.pyplot as pl
import matplotlib
from sklearn.kernel_ridge import KernelRidge
from vc_calculator import dict_averager
from fitter import *
from PIL import Image

opt = 5
archive5000 = '/home/hdft/Documents/DNN-Complete/CIFAR_New/'
archive1000 = '/home/hdft/Documents/DNN-Complete/1000run/'
archive11800 = '/home/hdft/Documents/DNN-Complete/MIT_New/'
plot_dir = '/home/hdft/Documents/DNN-Complete/DNN-PLOTS/Box_Plots/'

# nets_list7 = open("nets_list_MIT_10.txt", 'r')
# nets = nets_list7.readlines()
SET = ''

dicts = [{},{},{}]

for run, dic in zip([archive5000, archive1000, archive11800], xrange(len(dicts))):
    global_dic = {}
    global_plotters = {}
    counter = 0
    print run
    if run == archive5000:
        SET = 'CIFAR'
    elif run == archive1000:
        SET = 'MNIST'
    elif run == archive11800:
        SET = 'MIT'

    for file_list in os.listdir(run):
        if SET == 'CIFAR':
            nets_file = 'param_list_CIFAR_NEW.txt'
            if file_list.split('-')[4] == '10':
                nets_file = 'nets_list_CIFAR_new_10.txt'
            elif file_list.split('-')[4] == '20':
                nets_file = 'nets_list_CIFAR_new_20.txt'
            elif file_list.split('-')[4] == '40':
                nets_file = 'nets_list_CIFAR_new_40.txt'
            elif file_list.split('-')[4] == '80':
                nets_file = 'nets_list_CIFAR_new_80.txt'

        if SET == 'MNIST' :
            nets_file = 'param_list1000run.txt'
            if file_list.split('-')[4] == '10':
                nets_file = 'nets_list7.txt'
            elif file_list.split('-')[4] == '20':
                nets_file = 'nets_list20_7.txt'
            elif file_list.split('-')[4] == '40':
                nets_file = 'nets_list40_7.txt'
            elif file_list.split('-')[4] == '80':
                nets_file = 'nets_list80_7.txt'

        if SET == 'MIT' :
            nets_file = 'param_list_MIT_NEW.txt'
            if file_list.split('-')[4] == '10':
                nets_file = 'nets_list_MIT_new_10.txt'
            elif file_list.split('-')[4] == '20':
                nets_file = 'nets_list_MIT_new_20.txt'
            elif file_list.split('-')[4] == '40':
                nets_file = 'nets_list_MIT_new_40.txt'
            elif file_list.split('-')[4] == '80':
                nets_file = 'nets_list_MIT_new_80.txt'

        nets_foo = open(nets_file, 'r')
        nets_list = nets_foo.readlines()

        print file_list
        data_dic = {}
        print len(os.listdir(run + file_list + '/'))

        for d_file in os.listdir(run + file_list + '/'):
            if not d_file.startswith('.'):
                counter+=1
                data = numpy.load(run + file_list + '/' + d_file)
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
                    elif tup[3] == 1:
                        val_accuracy.append(tup[0])
                        val_error.append(tup[1])
                    elif tup[3] == 4:
                        continue
                    else:
                        print "error"

            dicts[dic][str(min(val_error)) + "+" + str(counter)] = [file_list + "+" + d_file.split('_')[3].split('.')[0] , val_error, val_accuracy, nets_list[int(d_file.split('_')[3].split('.')[0])], train_accuracy]
            global_dic[str(min(val_error)) + "+" + str(counter)] = [file_list + "+" + d_file.split('_')[3].split('.')[0] , val_error, val_accuracy]
        print len(global_dic)

medianprops = {'color': 'black'}
boxprops = {'color': 'black', 'linestyle': '-'}
whiskerprops = {'color': 'black', 'linestyle': '-'}
capprops = {'color': 'black', 'linestyle': '-'}

matplotlib.rcParams.update({'font.size': 4.7})
fig, axes = pl.subplots(1,3)
fig.set_size_inches(11, 5.0)

#subplot for accruacy histogram
#cifar histogram
# n_bins = [i/100.0 for i in range(0,300,2)]
acc_bins = [i for i in range(0,100, 2)]
axes[0].set_ylim([0,5000])
axes[0].set_title("CIFAR-10")
axes[0].set_xlabel("Classification Error %")
axes[0].set_ylabel("Number Of Networks")
axes[0].hist([100*(1.0-max(val[2])) for val in dicts[0].values()], bins = acc_bins, color=['grey'])
errors = [100*(1.0-max(val[2])) for val in dicts[0].values()]
print "CIFAR AVERAGE ERROR = " + str(float(sum(errors))/len(errors))
# pl.figure("Error Hist")
# pl.xlabel("Error Range")
# pl.ylabel("Number Of Networks")
# pl.hist([float(key.split('+')[0]) for key in global_dic.keys()], bins=n_bins)

#mnist histogram
# n_bins = [i/100.0 for i in range(0,300,2)]
acc_bins = [i for i in range(0,100, 2)]
axes[1].set_ylim([0,5000])
axes[1].set_title("MNIST-1000")
axes[1].set_xlabel("Classification Error %")
axes[1].set_ylabel("Number Of Networks")
axes[1].hist([100*(1.0-max(val[2])) for val in dicts[1].values()], bins = acc_bins, color=['grey'])
errors = [100*(1.0-max(val[2])) for val in dicts[1].values()]
print "MNIST AVERAGE ERROR = " + str(float(sum(errors))/len(errors))
#mitosis histogram
# n_bins = [i/100.0 for i in range(0,300,2)]
acc_bins = [i for i in range(0,100, 2)]
axes[2].set_ylim([0,5000])
axes[2].set_title("MITOSIS")
axes[2].set_xlabel("Classification Error %")
axes[2].set_ylabel("Number Of Networks")
axes[2].hist([100*(1.0-max(val[2])) for val in dicts[2].values()], bins = acc_bins, color=['grey'])
errors = [100*(1.0-max(val[2])) for val in dicts[2].values()]
print "MITOSIS AVERAGE ERROR = " + str(float(sum(errors))/len(errors))
#cifar error rate plot
# plots = dict_averager(dicts[0])
# for nets in plots:
#     axes[1,0].plot(nets)
# axes[1,0].set_xlabel("Epoch Number")
# axes[1,0].set_ylabel("Classification Error %")
#
# #mnist error rate plot
# plots = dict_averager(dicts[1])
# for nets in plots:
#     axes[1,1].plot(nets)
# axes[1,1].set_xlabel("Epoch Number")
# axes[1,1].set_ylabel("Classification Error %")
#
# #mitosis error rate plot
# plots = dict_averager(dicts[2])
# for nets in plots:
#     axes[1,2].plot(nets)
# axes[1,2].set_xlabel("Epoch Number")
# axes[1,2].set_ylabel("Classification Error %")

# global_sorted_list = []
#
# sorter_list = []
# for key in dicts[2].keys():
#     sorter_list.append([float(key.split('+')[0]), key])
#
# for key in sorted(sorter_list, key= lambda x: x[0], reverse=False):
#     global_sorted_list.append([dicts[2][key[1]], key[0]])
#
# print global_sorted_list[0]
#
# nets = open("param_list_MIT_NEW.txt", 'r').readlines()
#
# list_file_name = "DUMB" + run.split('/')[-2] + ".txt"
# pl.figure(run)
# pl.xlabel("Epoch Number")
# pl.ylabel("Validation Accuracy")
# with open(list_file_name, 'w+') as pam_file:
#     for ind in global_sorted_list[:opt]:
#         #pl.plot(ind[0][1])
#         print "Test" + str(ind[0][4][-1])
#         print max(ind[0][2])
#         print ind[0][0]
#         print nets[int(ind[0][0].split('+')[1])]
#         print ind[1]
#
#         # pam_file.write(nets[int(ind[0][0].split('+')[1])])
# print opt

fig.savefig(plot_dir + 'boxplot_hists_new_RUNS.png', dpi=300, format='png', bbox_inches='tight')
png2 = Image.open(plot_dir + 'boxplot_hists_new_RUNS.png')
png2.save(plot_dir + 'boxplot_hists_new_RUNS.tiff', compression='lzw')
pl.show()
