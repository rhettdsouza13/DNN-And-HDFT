from treelib import *
from math import *
from syntax_gen import *
import numpy
#MNIST PARAMETERS:::::
# #cost_br = 722610
# vc_max = 3500000
# max_layer = 500
# # vc_max = 30852000
# #vc_max = 32*32*100
# # cost_br = 181000
# inp_dim = 1024
# inp_dimx = 32
# inp_dimy = 32
# out_dim = 10
# sing_comp = 0
# relu_act = 0
# tanh_act = 0
# mul_cost_DN = 1.0
# mul_cost_CN = 1.0
#MNIST PARAMETERS END::::

#PATHOLOGY PARAMETERS:::::
#cost_br = 722610
# vc_max = 2630000
vc_max = 2960000
max_layer = 500
# vc_max = 30852000
#vc_max = 32*32*100
# cost_br = 181000
inp_dim = 4096
inp_dimx = 64
inp_dimy = 64
out_dim = 2
sing_comp = 0
relu_act = 0
tanh_act = 0
mul_cost_DN = 1.0
mul_cost_CN = 1.0
#PATHOLOGY PARAMETERS:::::

m_tree = Tree()

def power_gen(start,end):
    l_2_exp = [2**x for x in xrange(start,end)]
    return l_2_exp

def fact_val(start, end):
    l_val = [10*x for x in xrange(start,end)]
    return l_val

def fact_val_high(start, end):
    l_val = [40*x for x in xrange(start,end)]
    return l_val

def power_gen4(start,end):
    l_4_exp = [4**x for x in xrange(start,end)]
    return l_4_exp

m_tree.create_node("Root", 'r')
counter = 0



def tree_creater(parent, prev_dim, x, y, fc_dim, fl_flag, mp_flag, layer_num, weights):
    global counter
    counter+=1
    c_p=parent


    if fl_flag==0:

        for k in [5,7]:

            for output_ch in fact_val(1,2):

                iD = 'conv' + str(k) + '*' + str(k) + '_' + str(prev_dim) + "*" + str(output_ch) + '_n_' + str(counter)

                name = 'Convolution' + str(k) + '*' + str(k) + '_' + str(prev_dim) + "*" + str(output_ch) + '_n_' + str(counter)
                new_weights = weights + k*k*output_ch*prev_dim + output_ch
                vc_weights = weights + k*k*output_ch*prev_dim + x*y*out_dim*output_ch + output_ch + out_dim
                vc = vc_weights*(layer_num+2)*numpy.log2(vc_weights)
                # vc = new_weights*(layer_num+2)*numpy.log2(new_weights)

                if vc <= vc_max and layer_num < max_layer:
                    m_tree.create_node(name, iD, parent=c_p, data=[k,k,prev_dim,output_ch,x,y])
                    tree_creater(iD, output_ch, x, y, output_ch*x*y, 0, 0, layer_num+1, new_weights)


        if mp_flag==0:
            for k in power_gen(1,3):

                if k*4<x:

                    iD = 'mp' + str(k) + '*' + str(k) + '_' + str(x) + "*" + str(y)+ '_' + str(x/k) + "*" + str(y/k) + '_n_' + str(counter)

                    name = 'Max_Pooling' + str(k) + '*' + str(k) + '_' + str(x) + "*" + str(y)+ '_' + str(x/k) + "*" + str(y/k) + '_n_' + str(counter)

                    if layer_num<max_layer:
                        m_tree.create_node(name, iD, parent=c_p, data=[k,k,(x/k),(y/k), prev_dim])
                        # print "Pool"
                        tree_creater(iD, prev_dim, x/k, y/k, prev_dim*x*y/(k*k), 0, 1, layer_num, weights)


    for output_ch in fact_val(1,2):

        iD = 'fc' + '_' + str(fc_dim) + "*" + str(output_ch) + '_n_' + str(counter)

        name = 'Dense' + '_' + str(fc_dim) + "*" + str(output_ch) + '_n_' + str(counter)

        new_weights = weights + output_ch*fc_dim + output_ch
        vc_weights = weights + output_ch*fc_dim + output_ch*out_dim + output_ch + out_dim
        vc = vc_weights*(layer_num+2)*numpy.log2(vc_weights)

        if vc <= vc_max and layer_num<max_layer:
            m_tree.create_node(name, iD, parent=c_p, data=[fc_dim, output_ch])
            # print "Dense"
            tree_creater(iD, output_ch, x, y, output_ch, 1, 0, layer_num+1, new_weights)


    out_vc_weights = weights + fc_dim*out_dim + out_dim
    out_vc = out_vc_weights*(layer_num+1)*numpy.log2(out_vc_weights)
    # if out_vc<=vc_max and layer_num<=max_layer:
    counter+=1
    iD = 'out' + '_' + str(fc_dim) + "*" + str(out_dim)+'_n_' + str(counter)
    name = 'Out' + '_' + str(fc_dim) + "*" + str(out_dim)+ '_n_' + str(counter)
    m_tree.create_node(name, iD, parent=c_p, data=[fc_dim, out_dim, out_vc])



prev_dim=3
tree_creater('r', prev_dim, inp_dimx, inp_dimy, prev_dim*inp_dimx*inp_dimy, 0, 1, 0, 0)
m_tree.show(line_type='ascii-emv')
syntax_generator(m_tree, inp_dimx, inp_dimy)
