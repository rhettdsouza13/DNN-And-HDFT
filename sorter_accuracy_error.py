import os
import numpy
import matplotlib.pyplot as pl

opt = 5
archive5000 = '/home/hdft/Documents/DNN-Complete/5000-CIFAR-Run/'
# archive100 = '/home/hdft/Documents/DNN-Complete/100Run/'
# archive1000 = '/home/hdft/Documents/DNN-Complete/1000run/'

nets_list7 = open("nets_list_CIFAR_7.txt", 'r')
nets = nets_list7.readlines()

for run in [archive5000]:
    global_dic = {}
    global_plotters = {}
    for file_list in os.listdir(run):
        data_dic = {}
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
            #global_plotters[file_list + "+" + d_file.split('_')[3].split('.')[0]] = val_error
            data_dic[min(val_error)] = [file_list + "+" + d_file.split('_')[3].split('.')[0] , val_error, val_accuracy]
            global_dic[min(val_error)] = [file_list + "+" + d_file.split('_')[3].split('.')[0] , val_error, val_accuracy]
    n_bins = [i/100.0 for i in range(0,800,2)]
    acc_bins = [i/100.0 for i in range(0,60, 2)]
    pl.figure("Accuracy Hist")
    pl.hist([max(val[2]) for val in global_dic.values()], bins = acc_bins)
    pl.figure("Error Hist")
    pl.hist(global_dic.keys(), bins=n_bins)
    #sorted_list = []
    global_sorted_list = []

    # for key in sorted(data_dic.iterkeys()):
    #     #print data_dic[key], key
    #     sorted_list.append(data_dic[key])

    for key in sorted(global_dic.iterkeys()):
        #print data_dic[key], key
        global_sorted_list.append([global_dic[key], key])

        #print sorted_list[:opt]

        # for ind in sorted_list[:opt]:
        #     print nets[int(ind.split('+')[1])]

    print global_sorted_list[:opt]
    list_file_name = "DUMP" + run.split('/')[-2] + ".txt"
    pl.figure(run)
    pl.xlabel("Epoch Number")
    pl.ylabel("Validation Accuracy")
    with open(list_file_name, 'w+') as pam_file:
        for ind in global_sorted_list[:opt]:
            pl.plot(ind[0][2])
            print ind[0][0]
            print nets[int(ind[0][0].split('+')[1])]
            print ind[1]

            # pam_file.write(nets[int(ind[0][0].split('+')[1])])
pl.show()
