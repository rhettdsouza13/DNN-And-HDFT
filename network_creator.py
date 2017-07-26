from treelib import *
from math import *
from syntax_gen import *

cost_br = 5000
inp_dim = 1024
inp_dimx = 32
inp_dimy = 32
out_dim = 10
sing_comp = 1.0/1000
relu_act = 1.0/1000
tanh_act = 1.0/500
mul_cost_DN = 1.0/50
mul_cost_CN = 1.0/250

m_tree = Tree()

def power_gen(start,end):
    l_2_exp = [2**x for x in xrange(start,end)]
    return l_2_exp


m_tree.create_node("Root", 'r')
counter = 0

def tree_creater(parent, prev_dim, x, y, fc_dim, cost, fl_flag, mp_flag):
    global counter
    counter+=1
    c_p=parent

    if fl_flag==0:

        for k in power_gen(2,4):

            for output_ch in power_gen(4,15):

                iD = 'conv' + str(k) + '*' + str(k) + '_' + str(prev_dim) + "*" + str(output_ch) + '_n_' + str(counter)

                name = 'Convolution' + str(k) + '*' + str(k) + '_' + str(prev_dim) + "*" + str(output_ch) + '_n_' + str(counter)


                next_cost = cost - (float(k*k*prev_dim*output_ch*x*y)*mul_cost_CN) - (x*y*tanh_act*output_ch)

                if next_cost-(output_ch*out_dim*x*y*mul_cost_DN)>0:

                    m_tree.create_node(name, iD, parent=c_p, data=[k,k,prev_dim,output_ch,x,y])
                    tree_creater(iD, output_ch, x, y, output_ch*x*y, next_cost, 0, 0)


        if mp_flag==0:
            for k in power_gen(1,2):

                if k*4<x:

                    iD = 'mp' + str(k) + '*' + str(k) + '_' + str(x) + "*" + str(y)+ '_' + str(x/k) + "*" + str(y/k) + '_n_' + str(counter)

                    name = 'Max_Pooling' + str(k) + '*' + str(k) + '_' + str(x) + "*" + str(y)+ '_' + str(x/k) + "*" + str(y/k) + '_n_' + str(counter)

                    next_cost = cost - (((x/k)**2)*sing_comp*prev_dim)

                    if next_cost-(prev_dim*out_dim*(x/k)*(y/k)*mul_cost_DN)>0:

                        m_tree.create_node(name, iD, parent=c_p, data=[k,k,(x/k),(y/k), prev_dim])
                        tree_creater(iD, prev_dim, x/k, y/k, prev_dim*x*y/(k*k), next_cost, 0, 0)


    for output_ch in power_gen(8,15):

        iD = 'fc' + '_' + str(fc_dim) + "*" + str(output_ch) + '_n_' + str(counter)

        name = 'Dense' + '_' + str(fc_dim) + "*" + str(output_ch) + '_n_' + str(counter)

        next_cost = cost - (fc_dim*output_ch*mul_cost_DN) - (tanh_act*output_ch)

        if next_cost-(output_ch*out_dim*mul_cost_DN)>0:
            m_tree.create_node(name, iD, parent=c_p, data=[fc_dim, output_ch])
            tree_creater(iD, output_ch, x, y, output_ch, next_cost, 1, 0)



    counter+=1
    iD = 'out' + '_' + str(fc_dim) + "*" + str(out_dim)+'_n_' + str(counter)
    name = 'Out' + '_' + str(fc_dim) + "*" + str(out_dim)+ '_n_' + str(counter)
    m_tree.create_node(name, iD, parent=c_p, data=[fc_dim,out_dim])



prev_dim=1
tree_creater('r', prev_dim, inp_dimx, inp_dimy, prev_dim*inp_dimx*inp_dimy, cost_br, 0, 1)
m_tree.show()
syntax_generator(m_tree, inp_dimx, inp_dimy)
