import os
import numpy
import pipes

net_f_10 = open("nets_list_PATH_10.txt", 'r')
net_10 = net_f_10.readlines()
net_f_20 = open("nets_list_PATH_20.txt", 'r')
net_20 = net_f_20.readlines()
net_f_40 = open("nets_list_PATH_40.txt", 'r')
net_40 = net_f_40.readlines()
net_f_80 = open("nets_list_PATH_80.txt", 'r')
net_80 = net_f_80.readlines()

# net_num = 0
# for net in net_20:
#     print net_num
#     os.system("python trainer_tester_dropping100.py %s %d %d %d" % (pipes.quote(net), net_num, 14, 20))
#     net_num += 1
#
#
# net_num = 0
# for net in net_40:
#     print net_num
#     os.system("python trainer_tester_dropping100.py %s %d %d %d" % (pipes.quote(net), net_num, 15, 40))
#     net_num += 1
#
#
# net_num = 0
# for net in net_80:
#     print net_num
#     os.system("python trainer_tester_dropping100.py %s %d %d %d" % (pipes.quote(net), net_num, 16, 80))
#     net_num += 1

net_num = 0
for net in net_10:
    print net_num
    os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 206, 10))
    net_num += 1

net_num = 0
for net in net_20:
    print net_num
    os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 207, 20))
    net_num += 1


net_num = 0
for net in net_40:
    print net_num
    os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 208, 40))
    net_num += 1


net_num = 0
for net in net_80:
    print net_num
    os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 209, 80))
    net_num += 1

net_f_80.close()
net_f_40.close()
net_f_20.close()
net_f_10.close()
