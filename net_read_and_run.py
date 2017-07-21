import tensorflow as tf
import numpy
from functions import *
from parser import *
import sys

net_syn = sys.argv[1].split('|')
ch_list = []
fn_list=[]
print net_syn

counter = 0

for notat in net_syn:
    feat = notat.split(',')
    if feat[0] == 'conv':
        fn_list.append(['conv', feat[1], int(feat[2])])
    elif feat[0] == 'max_pooling':
        fn_list.append(['max_pooling', feat[1], int(feat[2])])
    elif feat[0] == 'full':
        fn_list.append(['full', feat[1]])

    else:
        ch_list.append(map(int, feat))


print ch_list
print fn_list

weights = {}
bias = {}

for ind in xrange(len(ch_list)-1):
    if fn_list[ind][0]=='conv':
        print 'here'
        prev_dim = ch_list[ind][2]
        output_ch = ch_list[ind+1][2]
        W_var_name = 'W' + str(ind)
        b_var_name = 'b' + str(ind)
        weights[W_var_name] = weight_initializer([fn_list[ind][2], fn_list[ind][2], prev_dim, output_ch])
        bias[b_var_name] = bias_initializer([output_ch])
print weights
print bias
