import os
import sys
import numpy
import pipes

file_list = '/home/hdft/Documents/DNN-Data-Run-9-800-200-drop/'

#data_dic = {}
flagged = []
net_file = open("nets_list7.txt", 'r')
nets = net_file.readlines()
iterator_e = 0
frac = [10,5,2]
for epoch_pnt in [i*5 for i in [1,3,5,20]]:

    net_num = 0

    for net in nets:
        print net_num
        if epoch_pnt == 5:
            os.system("python trainer_tester_dropping.py %s %d %d" % (pipes.quote(net), net_num, epoch_pnt))
        else:
            if net_num in flagged[:len(flagged)/frac[iterator_e]]:
                os.system("python trainer_tester_dropping.py %s %d %d" % (pipes.quote(net), net_num, epoch_pnt))
        net_num += 1
    data_dic = {}
    for d_file in os.listdir(file_list):

        data = numpy.load(file_list + d_file)
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
        print len(val_error)
        if len(val_error) == epoch_pnt:
            val_error[epoch_pnt-1]
            data_dic[val_error[epoch_pnt-1]] = int(d_file.split('_')[3].split('.')[0])

    flagged = []
    print data_dic
    for key in sorted(data_dic.iterkeys()):
        #print data_dic[key], key
        flagged.append(data_dic[key])

    print flagged[len(flagged)/frac[iterator_e]:]
    iterator_e += 1
