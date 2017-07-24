import tensorflow as tf
import numpy
from functions import *
from parser import *
import sys
from operator import mul

def set_variables(network):
    net_syn = network.split('|')
    ch_list = []
    fn_list=[]
    print net_syn


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
    counter=0
    for ind in xrange(len(ch_list)-1):

        if fn_list[ind][0]=='conv':

            prev_dim = ch_list[ind][2]
            output_ch = ch_list[ind+1][2]
            W_var_name = 'W' + str(ind)
            b_var_name = 'b' + str(ind)
            weights[W_var_name] = weight_initializer([fn_list[ind][2], fn_list[ind][2], prev_dim, output_ch])
            bias[b_var_name] = bias_initializer([output_ch])
            counter+=1

        if fn_list[ind][0]=='full':

            prev_dim = reduce(mul, ch_list[ind])
            output_ch = reduce(mul, ch_list[ind+1])
            W_var_name = 'W' + str(ind)
            b_var_name = 'b' + str(ind)
            weights[W_var_name] = weight_initializer([prev_dim, output_ch])
            bias[b_var_name] = bias_initializer([output_ch])
            counter+=1

    return weights, bias, fn_list, ch_list

w,b,fn,ch = set_variables('16,16,1|conv,tanh,2|16,16,16|max_pooling,identity,2|8,8,16|conv,tanh,2|8,8,8|conv,tanh,2|8,8,8|full,tanh|1,1,10')
