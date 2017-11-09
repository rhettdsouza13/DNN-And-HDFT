import os
import numpy

opt = 20
archive500 = '/home/hdft/Documents/DNN-Complete/500run/'
nets_list7 = open("nets_list7.txt", 'r')
nets = nets_list7.readlines()
global_dic = {}
for file_list in os.listdir(archive500):
    data_dic = {}
    for d_file in os.listdir(archive500 + file_list + '/'):

        data = numpy.load(archive500 + file_list + '/' + d_file)
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

        data_dic[min(val_error)] = file_list + "+" + d_file.split('_')[3].split('.')[0]
        global_dic[min(val_error)] = file_list + "+" + d_file.split('_')[3].split('.')[0]

    sorted_list = []
    global_sorted_list = []

    for key in sorted(data_dic.iterkeys()):
        #print data_dic[key], key
        sorted_list.append(data_dic[key])

    for key in sorted(global_dic.iterkeys()):
        #print data_dic[key], key
        global_sorted_list.append(global_dic[key])

    #print sorted_list[:opt]

    # for ind in sorted_list[:opt]:
    #     print nets[int(ind.split('+')[1])]

print global_sorted_list[:opt]

for ind in global_sorted_list[:opt]:
    print nets[int(ind.split('+')[1])]
