import os
import numpy
import matplotlib.pyplot as pl
import matplotlib

opt = 5
archive5000 = '/home/hdft/Documents/DNN-Complete/5000-CIFAR-Run/'
archive1000 = '/home/hdft/Documents/DNN-Complete/1000run/'
archive11800 = '/home/hdft/Documents/DNN-Complete/11800-MIT-Run/'

nets_list7 = open("nets_list_MIT_10.txt", 'r')
nets = nets_list7.readlines()


dicts = [{},{},{}]

for run, dic in zip([archive5000, archive1000, archive11800], xrange(len(dicts))):
    global_dic = {}
    global_plotters = {}
    counter = 0
    print run
    for file_list in os.listdir(run):
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

            dicts[dic][str(min(val_error)) + "+" + str(counter)] = [file_list + "+" + d_file.split('_')[3].split('.')[0] , val_error, val_accuracy]
            global_dic[str(min(val_error)) + "+" + str(counter)] = [file_list + "+" + d_file.split('_')[3].split('.')[0] , val_error, val_accuracy]
        print len(global_dic)

medianprops = {'color': 'black'}
boxprops = {'color': 'black', 'linestyle': '-'}
whiskerprops = {'color': 'black', 'linestyle': '-'}
capprops = {'color': 'black', 'linestyle': '-'}

matplotlib.rcParams.update({'font.size': 4.7})
fig, axes = pl.subplots(2,3)
fig.set_size_inches(11, 15.0)

#subplot for accruacy histogram
#cifar histogram
# n_bins = [i/100.0 for i in range(0,300,2)]
acc_bins = [i/100.0 for i in range(0,100, 2)]
axes[0,0].set_title("CIFAR-10")
axes[0,0].set_xlabel("Accuracy")
axes[0,0].set_ylabel("Number Of Networks")
axes[0,0].hist([max(val[2]) for val in dicts[0].values()], bins = acc_bins, color=['grey'])
# pl.figure("Error Hist")
# pl.xlabel("Error Range")
# pl.ylabel("Number Of Networks")
# pl.hist([float(key.split('+')[0]) for key in global_dic.keys()], bins=n_bins)

#mnist histogram
# n_bins = [i/100.0 for i in range(0,300,2)]
acc_bins = [i/100.0 for i in range(0,100, 2)]
axes[0,1].set_title("MNIST")
axes[0,1].set_xlabel("Accuracy")
axes[0,1].set_ylabel("Number Of Networks")
axes[0,1].hist([max(val[2]) for val in dicts[1].values()], bins = acc_bins, color=['grey'])

#mitosis histogram
# n_bins = [i/100.0 for i in range(0,300,2)]
acc_bins = [i/100.0 for i in range(70,100, 2)]
axes[0,2].set_title("MITOSIS")
axes[0,2].set_xlabel("Accuracy")
axes[0,2].set_ylabel("Number Of Networks")
axes[0,2].hist([max(val[2]) for val in dicts[2].values()], bins = acc_bins, color=['grey'])


# global_sorted_list = []
#
# sorter_list = []
# for key in global_dic.keys():
#     sorter_list.append([float(key.split('+')[0]), key])
#
# for key in sorted(sorter_list, key= lambda x: x[0]):
#     global_sorted_list.append([global_dic[key[1]], key[0]])
#
# print global_sorted_list[0]

# list_file_name = "opt_list_" + run.split('/')[-2] + ".txt"
# pl.figure(run)
# pl.xlabel("Epoch Number")
# pl.ylabel("Validation Accuracy")
# with open(list_file_name, 'w+') as pam_file:
#     for ind in global_sorted_list[:opt]:
#         pl.plot(ind[0][1])
#         print max(ind[0][2])
#         print ind[0][0]
#         print nets[int(ind[0][0].split('+')[1])]
#         print ind[1]
#
#         pam_file.write(nets[int(ind[0][0].split('+')[1])])
# print opt

pl.show()
