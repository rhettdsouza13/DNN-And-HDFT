import os
import numpy
import matplotlib.pyplot as pl
import matplotlib
from PIL import Image
from vc_calculator import *

archive500_MNIST = '/home/hdft/Documents/DNN-Complete/500run/'
archive1000_MNIST = '/home/hdft/Documents/DNN-Complete/1000run/'
archive100_MNIST = '/home/hdft/Documents/DNN-Complete/100Run/'
plot_dir = '/home/hdft/Documents/DNN-Complete/DNN-PLOTS/Box_Plots/'
SET = 'MNIST'

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
c_100 = 0
c_500 = 0
c_1000 = 0
for run in [archive100_MNIST, archive500_MNIST, archive1000_MNIST]:
    print "On: " + run
    #print c_100, MNIST_f_c, MIT_f_c

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

        if SET == 'MIT' :
            if file_list.split('-')[4] == '10':
                nets_file = 'nets_list_MIT_10.txt'
            elif file_list.split('-')[4] == '20':
                nets_file = 'nets_list_MIT_20.txt'
            elif file_list.split('-')[4] == '40':
                nets_file = 'nets_list_MIT_40.txt'
            elif file_list.split('-')[4] == '80':
                nets_file = 'nets_list_MIT_80.txt'

        nets_foo = open(nets_file, 'r')
        nets_list = nets_foo.readlines()

        #continuing with data analysis
        for d_file in os.listdir(run + file_list + '/'):

            if run == archive100_MNIST:
                c_100 += 1
            elif run == archive500_MNIST:
                c_500 += 1
            elif run == archive1000_MNIST:
                c_1000 += 1

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
            # depth = len(net_parts) - 2
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
            depth = conv+mp+full
            # if int(key.split('+')[1]) == 1451:
            #     print network
            #     print depth
            plotY_er.append(min(data_dic[key][0]))
            plotY_acc.append(max(data_dic[key][1]))
            # if depth == 1:
            #     print depth
            plotX_vc.append(vc)
            plotX_mp.append(mp)
            plotX_depth.append(depth)
            plotX_conv.append(conv)
            plotX_fc.append(full)
    print c_100, c_500, c_1000

medianprops = {'color': 'black'}
boxprops = {'color': 'black', 'linestyle': '-'}
whiskerprops = {'color': 'black', 'linestyle': '-'}
capprops = {'color': 'black', 'linestyle': '-'}

plotY_error_rate = [100*(1-x) for x in plotY_acc]
plotX_vc_log = numpy.log2(plotX_vc)

#basic figure subplot division and param setting
matplotlib.rcParams.update({'font.size': 4.7})
fig, axes = pl.subplots(5,3)
fig.set_size_inches(11, 15.0)


#cifar subplotVC
N = max(plotX_vc_log[:c_100]) - min(plotX_vc_log[:c_100])
bins, pos = binner(plotX_vc_log[:c_100], plotY_error_rate[:c_100], N/2, vc_flag = 1)
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i]) + ' - ' + str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))

axes[0,0].set_ylim([0,100])
axes[0,0].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)

axes[0,0].set_title('MNIST-100', fontsize=7)
axes[0,0].set_xlabel(r'$\mathregular{VC-Dimension(log_2 Scale)}$', fontsize=5.5)
axes[0,0].set_ylabel('Classification Error %', fontsize=5.5)

#MNIST subplotVC
c_500 += c_100
N = max(plotX_vc_log[c_100:c_500]) - min(plotX_vc_log[c_100:c_500])

print "VALUE OF N : " + str(N)
print min(plotX_vc_log[c_100:c_500])
print max(plotX_vc_log[c_100:c_500])
bins, pos = binner(plotX_vc_log[c_100:c_500], plotY_error_rate[c_100:c_500], N/2, vc_flag = 1)
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i]) + ' - ' + str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))

axes[0,1].set_ylim([0,100])
axes[0,1].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)

axes[0,1].set_title('MNIST-500', fontsize=7)
axes[0,1].set_xlabel(r'$\mathregular{VC-Dimension(log_2 Scale)}$', fontsize=5.5)
axes[0,1].set_ylabel('Classification Error %', fontsize=5.5)

#MIT subplotVC
c_1000 += c_500
N = max(plotX_vc_log[c_500:c_1000]) - min(plotX_vc_log[c_500:c_1000])
print "VALUE OF N : " + str(N)
bins, pos = binner(plotX_vc_log[c_500:c_1000], plotY_error_rate[c_500:c_1000], N/2, vc_flag = 1)
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i]) + ' - ' + str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))

axes[0,2].set_ylim([0,100])
axes[0,2].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)

axes[0,2].set_title('MNIST-1000', fontsize=7)
axes[0,2].set_xlabel(r'$\mathregular{VC-Dimension(log_2 Scale)}$', fontsize=5.5)
axes[0,2].set_ylabel('Classification Error %', fontsize=5.5)

#depth plotting
#CIFAR DEPTH SUBPLOT
N=8
bins, pos = binner(plotX_depth[:c_100], plotY_error_rate[:c_100], N, fl_flag=1)
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == 1:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]+1) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))

axes[1,0].set_ylim([0,100])
axes[1,0].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[1,0].set_xlabel('Depth', fontsize=5.5)
axes[1,0].set_ylabel('Classification Error %', fontsize=5.5)

#MNIST DEPTH SUBPLOT
N=8
bins, pos = binner(plotX_depth[c_100:c_500], plotY_error_rate[c_100:c_500], N, fl_flag=1)
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == 1:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]+1) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))
axes[1,1].set_ylim([0,100])
axes[1,1].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[1,1].set_xlabel('Depth', fontsize=5.5)
axes[1,1].set_ylabel('Classification Error %', fontsize=5.5)

#MITOSIS DEPTH SUBPLOT
N=8
bins, pos = binner(plotX_depth[c_500:c_1000], plotY_error_rate[c_500:c_1000], N, fl_flag=1)
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == 1:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]+1) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))

axes[1,2].set_ylim([0,100])
axes[1,2].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[1,2].set_xlabel('Depth', fontsize=5.5)
axes[1,2].set_ylabel('Classification Error %', fontsize=5.5)

#conv plotting
#CIFAR conv SUBPLOT
N = max(plotX_conv[:c_100]) - min(plotX_conv[:c_100])
bins, pos = binner(plotX_conv[:c_100], plotY_error_rate[:c_100], N)

axes[2,0].set_ylim([-100,100])
axes[2,0].boxplot(bins, labels=pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[2,0].set_xlabel('Number Of Convolutional Layers', fontsize=5.5)
axes[2,0].set_ylabel('Classification Error %', fontsize=5.5)

#MNIST CONV SUBPLOT
N = max(plotX_conv[c_100:c_500]) - min(plotX_conv[c_100:c_500])
bins, pos = binner(plotX_conv[c_100:c_500], plotY_error_rate[c_100:c_500], N)
axes[2,1].set_ylim([0,100])
axes[2,1].boxplot(bins, labels=pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[2,1].set_xlabel('Number Of Convolutional Layers', fontsize=5.5)
axes[2,1].set_ylabel('Classification Error %', fontsize=5.5)

#MITOSIS CONV SUBPLOT
N = max(plotX_conv[c_500:c_1000]) - min(plotX_conv[c_500:c_1000])
bins, pos = binner(plotX_conv[c_500:c_1000], plotY_error_rate[c_500:c_1000], N)
axes[2,2].set_ylim([0,100])
axes[2,2].boxplot(bins, labels=pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[2,2].set_xlabel('Number Of Convolutional Layers', fontsize=5.5)
axes[2,2].set_ylabel('Classification Error %', fontsize=5.5)

#FC plotting
#CIFAR FC SUBPLOT
N = 6
bins, pos = binner(plotX_fc[:c_100], plotY_error_rate[:c_100], N, fl_flag=1)
print pos
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == 1:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]+1) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))
axes[3,0].set_ylim([0,100])
axes[3,0].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[3,0].set_xlabel('Number Of Fully Connected Layers', fontsize=5.5)
axes[3,0].set_ylabel('Classification Error %', fontsize=5.5)

#MNIST FC SUBPLOT
N = 6
bins, pos = binner(plotX_fc[c_100:c_500], plotY_error_rate[c_100:c_500], N, fl_flag=1)
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == 1:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]+1) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))
axes[3,1].set_ylim([0,100])
axes[3,1].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[3,1].set_xlabel('Number Of Fully Connected Layers', fontsize=5.5)
axes[3,1].set_ylabel('Classification Error %', fontsize=5.5)

#MITOSIS FC SUBPLOT
N = 6
bins, pos = binner(plotX_fc[c_500:c_1000], plotY_error_rate[c_500:c_1000], N, fl_flag=1)
labels_pos = []
for i in xrange(len(pos)):
    if i == 0:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == 1:
        labels_pos.append(str(pos[i+1]))
        continue
    if i == len(pos) - 1:
        labels_pos.append(str(pos[i]+1) + ' - ')
        continue
    labels_pos.append(str(pos[i]+1) + ' - ' + str(pos[i+1]))
axes[3,2].set_ylim([0,100])
axes[3,2].boxplot(bins, labels=labels_pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[3,2].set_xlabel('Number Of Fully Connected Layers', fontsize=5.5)
axes[3,2].set_ylabel('Classification Error %', fontsize=5.5)

#MP plotting
#CIFAR MP SUBPLOT
N = max(plotX_mp[:c_100]) - min(plotX_mp[:c_100])
bins, pos = binner(plotX_mp[:c_100], plotY_error_rate[:c_100], N)
axes[4,0].set_ylim([0,100])
axes[4,0].boxplot(bins, labels=pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[4,0].set_xlabel('Number Of Max-Pooling Layers', fontsize=5.5)
axes[4,0].set_ylabel('Classification Error %', fontsize=5.5)

#MNIST MP SUBPLOT
N = max(plotX_mp[c_100:c_500]) - min(plotX_mp[c_100:c_500])
bins, pos = binner(plotX_mp[c_100:c_500], plotY_error_rate[c_100:c_500], N)
axes[4,1].set_ylim([0,100])
axes[4,1].boxplot(bins, labels=pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[4,1].set_xlabel('Number Of Max-Pooling Layers', fontsize=5.5)
axes[4,1].set_ylabel('Classification Error %', fontsize=5.5)

#MITOSIS MP SUBPLOT
N = max(plotX_mp[c_500:c_1000]) - min(plotX_mp[c_500:c_1000])
bins, pos = binner(plotX_mp[c_500:c_1000], plotY_error_rate[c_500:c_1000], N)
axes[4,2].set_ylim([0,100])
axes[4,2].boxplot(bins, labels=pos,  medianprops=medianprops, sym='',
           boxprops=boxprops,
           whiskerprops=whiskerprops,
           capprops=capprops)
axes[4,2].set_xlabel('Number Of Max-Pooling Layers', fontsize=5.5)
axes[4,2].set_ylabel('Classification Error %', fontsize=5.5)
#
# N=max(plotX_conv) - min(plotX_conv)
# bins, pos = binner(plotX_conv, plotY_error_rate, N)
# pl.figure("CONV_BOX")
# pl.boxplot(bins, labels=pos, sym='')
#
# N=max(plotX_mp) - min(plotX_mp)
# bins, pos = binner(plotX_mp, plotY_er, N)
# pl.figure("MP_BOX")
# pl.boxplot(bins, labels=pos, sym='')
# #print plotX_mp
#
# N=max(plotX_fc) - min(plotX_fc)
# bins, pos = binner(plotX_fc, plotY_er, N)
# pl.figure("FC_BOX")
# pl.boxplot(bins, labels=pos, sym='')

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
#
# fig.savefig(plot_dir + 'boxplot3.png', dpi=300, format='png')
# png2 = Image.open(plot_dir + 'boxplot3.png')
# png2.save(plot_dir + 'boxplot3_MNIST.tiff', compression='lzw')
pl.show()
