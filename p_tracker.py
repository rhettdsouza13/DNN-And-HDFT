import os
import numpy
import pipes

net_f_20 = open("nets_list20_7.txt", 'r')
net_20 = net_f_20.readlines()
net_f_40 = open("nets_list40_7.txt", 'r')
net_40 = net_f_40.readlines()
net_f_80 = open("nets_list80_7.txt", 'r')
net_80 = net_f_80.readlines()

net_num = 0
for net in net_20:
    print net_num
    os.system("python trainer_tester_dropping.py %s %d %d %d" % (pipes.quote(net), net_num, 11, 20))
    net_num += 1
    

net_num = 0
for net in net_40:
    print net_num
    os.system("python trainer_tester_dropping.py %s %d %d %d" % (pipes.quote(net), net_num, 12, 40))
    net_num += 1


net_num = 0
for net in net_80:
    print net_num
    os.system("python trainer_tester_dropping.py %s %d %d %d" % (pipes.quote(net), net_num, 13, 80))
    net_num += 1


net_f_80.close()
net_f_40.close()
net_f_20.close()