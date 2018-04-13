import os
import numpy
import matplotlib.pyplot as pl
import matplotlib
from PIL import Image
from vc_calculator import *

archive5000 = '/home/hdft/Documents/DNN-Complete/5000-CIFAR-Run/'
plot_dir = '/home/hdft/Documents/DNN-Complete/DNN-PLOTS/Box_Plots/'
SET = 'CIFAR'

plotX_depth = []
#plotY_depth = []
plotX_conv = []
#plotY_conv = []
plotX_fc = []
#plotY_fc = []
plotX_mp = []
plotX_vc = []
plotY_er = []
plotY_acc = []

for run in [archive5000]:
    for file_list in os.listdir(run):
        data_dic = {}
        #opening network list
        if SET == 'CIFAR':
            if file_list.split('-')[4] == '10':
                nets_file = 'nets_list_CIFAR_7.txt'
            elif file_list.split('-')[4] == '20':
                nets_file = 'nets_list_CIFAR_20_7.txt'
            elif file_list.split('-')[4] == '40':
                nets_file = 'nets_list_CIFAR_40_7.txt'
            elif file_list.split('-')[4] == '80':
                nets_file = 'nets_list_CIFAR_80_7.txt'

        if SET == 'MNIST' :
            if file_list.split('-')[4] == '10':
                nets_file = 'nets_list7.txt'
            elif file_list.split('-')[4] == '20':
                nets_file = 'nets_list20_7.txt'
            elif file_list.split('-')[4] == '40':
                nets_file = 'nets_list40_7.txt'
            elif file_list.split('-')[4] == '80':
                nets_file = 'nets_list80_7.txt'

        nets_foo = open(nets_file, 'r')
        nets_list = nets_foo.readlines()

        #continuing with data analysis
        for d_file in os.listdir(run + file_list + '/'):

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
                if tup[3] == 1:
                    val_accuracy.append(tup[0])
                    val_error.append(tup[1])
            data_dic[file_list + "+" + d_file.split('_')[3].split('.')[0]] = [val_error, val_accuracy]


        for key in data_dic.iterkeys():
            network = nets_list[int(key.split('+')[1])]
            net_parts = network.split('|')
            depth = 0
            depth = len(net_parts) - 2
            vc = 0
            vc = calc_vc(network)
            conv = 0
            full = 0
            mp = 0
            for part in net_parts:
                if 'conv' in part:
                    conv += 1
                if 'full' in part:
                    full += 1
                if 'max_pooling' in part:
                    mp += 1

            plotY_er.append(min(data_dic[key][0]))
            plotY_acc.append(max(data_dic[key][1]))

            plotX_vc.append(vc)
            plotX_mp.append(mp)
            plotX_depth.append(depth)
            plotX_conv.append(conv)
            plotX_fc.append(full)



plotX_vc_log = numpy.log2(plotX_vc)
N = max(plotX_vc_log) - min(plotX_vc_log)
bins, pos = binner(plotX_vc_log, plotY_er, N/2)
#print bins[len(bins) - 1]
labels_pos = []
for i in xrange(len(pos)):
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]) + ' - ')
        continue
    labels_pos.append(str(pos[i]) + ' - ' + str(pos[i+1]))

matplotlib.rcParams.update({'font.size': 5})

pl.figure("VC_BOX")
pl.boxplot(bins, labels=labels_pos, sym='')
fig = pl.gcf()
fig.suptitle('Validation Accuracy v/s VC-Dimension', fontsize=7)
pl.xlabel('Range Of VC-Dimension', fontsize=5)
pl.ylabel('Accuracy', fontsize=5)
fig.set_size_inches(3.5, 2.75)
fig.savefig(plot_dir + 'vc_box.png', dpi=300)
img = Image.open(plot_dir + 'vc_box.png').convert('L')
img.save(plot_dir + 'vc_box_gs.png')
#print plotX_vc

N=20
bins, pos = binner(plotX_depth, plotY_er, N)
labels_pos = []
for i in xrange(len(pos)):
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]) + ' - ')
        continue
    labels_pos.append(str(pos[i]) + ' - ' + str(pos[i+1]))
pl.figure("DEPTH_BOX")
pl.boxplot(bins, labels=labels_pos, sym='')

N=max(plotX_conv) - min(plotX_conv)
bins, pos = binner(plotX_conv, plotY_er, N)
pl.figure("CONV_BOX")
pl.boxplot(bins, labels=pos, sym='')

N=max(plotX_mp) - min(plotX_mp)
bins, pos = binner(plotX_mp, plotY_er, N)
pl.figure("MP_BOX")
pl.boxplot(bins, labels=pos, sym='')
#print plotX_mp

N=max(plotX_fc) - min(plotX_fc)
bins, pos = binner(plotX_fc, plotY_er, N)
pl.figure("FC_BOX")
pl.boxplot(bins, labels=pos, sym='')

# pl.figure("Depth")
# pl.scatter(plotX_depth, plotY_er)
#
# pl.figure("Conv")
# pl.scatter(plotX_conv, plotY_er)
#
# pl.figure("Fully Connected")
# pl.scatter(plotX_fc, plotY_er)
#
# pl.figure("Max Pooling")
# pl.scatter(plotX_mp, plotY_er)
#
# pl.figure("DepthACC")
# pl.scatter(plotX_depth, plotY_acc)
#
# pl.figure("ConvACC")
# pl.scatter(plotX_conv, plotY_acc)
#
# pl.figure("Fully ConnectedACC")
# pl.scatter(plotX_fc, plotY_acc)
#
# pl.figure("Max PoolingACC")
# pl.scatter(plotX_mp, plotY_acc)
#
# pl.figure("VCACC")
# pl.scatter(plotX_vc, plotY_acc)
#
# pl.figure("VC")
# pl.scatter(plotX_vc, plotY_er)

# pl.figure("VC_BOX")
# pl.boxplot(plotX_vc, plotY_er)

pl.show()
