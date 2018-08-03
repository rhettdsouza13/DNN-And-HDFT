import os
import numpy
import pipes

net_f_10 = open("param_list_MIT_NEW.txt", 'r')
net_10 = net_f_10.readlines()
# net_f_20 = open("nets_list_MIT_new_20.txt", 'r')
# net_20 = net_f_20.readlines()
# net_f_40 = open("nets_list_MIT_new_40.txt", 'r')
# net_40 = net_f_40.readlines()
# net_f_80 = open("nets_list_MIT_new_80.txt", 'r')
# net_80 = net_f_80.readlines()

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
# #
# net_num = 0
# for net in net_80:
#     print net_num
#     os.system("python trainer_tester_dropping100.py %s %d %d %d" % (pipes.quote(net), net_num, 16, 80))
#     net_num += 1

# net_num = 0
# for net in net_10:
#     print net_num
#     if net_num <= 8215:
#         net_num+=1
#         continue
#     os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 214, 10))
#     net_num += 1

net_num = 0
for net in net_10:
    print net_num
    # if net_num <= 1961:
    #     net_num+=1
    #     continue
    os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 224, 10000))
    net_num += 1

# net_num = 0
# for net in net_40:
#     print net_num
#     os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 222, 40))
#     net_num += 1
# #
# net_num = 0
# for net in net_20:
#     print net_num
#     os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 221, 20))
#     net_num += 1
#
# net_num = 0
# for net in net_80:
#     print net_num
#     os.system("python trainer_tester.py %s %d %d %d" % (pipes.quote(net), net_num, 223, 80))
#     net_num += 1
#
# net_f_80.close()
# net_f_40.close()
# net_f_20.close()
net_f_10.close()
