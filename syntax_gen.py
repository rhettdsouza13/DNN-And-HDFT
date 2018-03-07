from treelib import *
import numpy
import matplotlib.pyplot as pl

def power_gen(start,end):
    l_2_exp = [2**x for x in xrange(start,end)]
    return l_2_exp

def syntax_generator(tree, x, y):
    paths = tree.paths_to_leaves()
    vcs = []
    with open('nets_list_PATH.txt', 'w+') as netfile:

        for path in paths:

            net = str(x) + ',' + str(y) + ',' + str(3) + '|'

            for node in path:

                if 'fc' in node:
                    dim = tree.get_node(node).data
                    net+='full,relu|'
                    net+='1,1,'+str(dim[1]) + '|'
                    continue
                if 'conv' in node:
                    dim = tree.get_node(node).data
                    net+='conv,relu,' + str(dim[0]) + '|' + str(dim[4]) + ',' + str(dim[5]) + ',' + str(dim[3]) + '|'
                    continue
                if 'mp' in node:
                    dim = tree.get_node(node).data
                    net+='max_pooling,identity,' + str(dim[0]) + '|' + str(dim[2]) + ',' + str(dim[3]) + ',' + str(dim[4]) + '|'
                    continue
                if 'out' in node:
                    dim = tree.get_node(node).data
                    net+='full,relu|'
                    net+='1,1,'+str(dim[1])
                    #net+='  ' + str(dim[2])
                    vcs.append(dim[2])
                    continue

            print net
            netfile.write(net + '\n')
#        print vcs
        vcs.sort()
        pl.scatter(range(len(vcs)), vcs)
        numpy.save("VCS", vcs)
        print len(paths)
        pl.show()
        print len(paths)

counter = 0
p_tree = Tree()
p_tree.create_node("Root", 'r')

def create_param_tree(parent, net_len):

    global counter
    c_p=parent
    counter += 1
    for output_ch in power_gen(5,9):
        if net_len > 0:
            name = str(output_ch) + '_net_len ' + str(net_len)
            iD = str(output_ch) + '_net_len ' + str(net_len) + '_' + str(counter)
            p_tree.create_node(name, iD, parent=c_p, data=[output_ch,1])
            create_param_tree(iD, net_len-1)


def param_iter(net):
    parts = net.split('|')
    print parts
    # xrange(((len(parts)-1)/2) - 1)
    iter=0
    func_ind = []
    m_flag = 0
    for func in parts:
        if iter==0 or iter==(len(parts)-1):
            iter+=1
            continue
        if 'max' in func:
            iter+=1
            m_flag=1
            continue
        if 'relu' in func:
            iter+=1
            continue
        if m_flag == 1:
            m_flag = 0
            iter+=1
            continue
        func_ind.append(iter)
        iter+=1

    create_param_tree('r', len(func_ind))
    p_tree.show()
    print len(func_ind)
    combos = p_tree.paths_to_leaves()
    # print combos

    with open("DUMP.txt", 'a+') as p_file:
        for path in combos:
            pos = 1
            cur = parts
            for ind in func_ind:
                # print path
                # print pos , ind
                dim = p_tree.get_node(path[pos]).data
                sec = cur[ind].split(',')
                sec[2] = str(dim[0])
                cur[ind] = ','.join(sec)
                pos+=1
                if 'max' in cur[ind+1]:
                    sec = cur[ind+2].split(',')
                    sec[2] = str(dim[0])
                    cur[ind+2] = ','.join(sec)

            cur = '|'.join(cur)
            print cur
            p_file.write(cur)
    print len(combos)
    print func_ind

# param_iter("32,32,3|conv,relu,5|32,32,10|max_pooling,identity,4|8,8,10|full,relu|1,1,10|full,relu|1,1,10|full,relu|1,1,10|full,relu|1,1,10|full,relu|1,1,10|full,relu|1,1,10")

# with open("opt_test_CIFAR_5000-CIFAR-Run.txt", 'r') as opt_file:
#     nets_list_opt = opt_file.readlines()
#     print nets_list_opt
#     for net in nets_list_opt:
#         counter = 0
#         p_tree = Tree()
#         p_tree.create_node("Root", 'r')
#         print net
#         param_iter(net)

def replace(net):

    parts = net.split('|')
    iter=0
    func_ind = []
    m_flag = 0
    for func in parts:
        if iter==0 or iter==(len(parts)-1):
            iter+=1
            continue
        if 'max' in func:
            iter+=1
            m_flag=1
            continue
        if 'relu' in func:
            iter+=1
            continue
        if m_flag == 1:
            m_flag = 0
            iter+=1
            continue
        func_ind.append(iter)
        iter+=1


    cur = parts
    sec = cur[0].split(',')
    sec[2] = str(3)
    cur[0] = ','.join(sec)
    if 'max' in cur[1]:
        sec = cur[2].split(',')
        sec[2] = str(3)
        cur[2] = ','.join(sec)
    for ind in func_ind:
        dim = 80
        sec = cur[ind].split(',')
        sec[2] = str(dim)
        cur[ind] = ','.join(sec)

        if 'max' in cur[ind+1]:
            sec = cur[ind+2].split(',')
            sec[2] = str(dim)
            cur[ind+2] = ','.join(sec)

    cur = '|'.join(cur)
    print cur
    return cur


# with open("nets_list80_7.txt", "r") as n_file, open("nets_list_CIFAR_80_7.txt", "w+") as out_file:
#     for prop in n_file.readlines():
#         out_net = replace(prop)
#         out_file.write(str(out_net))

#replace("32,32,1|max_pooling,identity,2|16,16,1|conv,relu,7|16,16,10|conv,relu,5|16,16,10|full,relu|1,1,10|full,relu|1,1,10")
# 32,32,1|max_pooling,identity,2|16,16,1|conv,relu,7|16,16,10|conv,relu,5|16,16,10|full,relu|1,1,10|full,relu|1,1,10
