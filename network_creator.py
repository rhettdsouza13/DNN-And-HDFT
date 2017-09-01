from treelib import *
from math import *
from syntax_gen import *

cost_br = 3000
inp_dim = 1024
inp_dimx = 32
inp_dimy = 32
out_dim = 10
sing_comp = 0
relu_act = 0
tanh_act = 0
mul_cost_DN = 1.0
mul_cost_CN = 1.0

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

        for k in [3,5,7]:

            for output_ch in power_gen(0,15):

                iD = 'conv' + str(k) + '*' + str(k) + '_' + str(prev_dim) + "*" + str(output_ch) + '_n_' + str(counter)

                name = 'Convolution' + str(k) + '*' + str(k) + '_' + str(prev_dim) + "*" + str(output_ch) + '_n_' + str(counter)


                next_cost = cost - (k*k*prev_dim*output_ch*mul_cost_CN) - (output_ch*x*y)

                #(x*y*relu_act*output_ch)

                if next_cost-(output_ch*out_dim*x*y*mul_cost_DN + out_dim) > 0:
                    m_tree.create_node(name, iD, parent=c_p, data=[k,k,prev_dim,output_ch,x,y])
                    print "Conv"
                    print next_cost
                    tree_creater(iD, output_ch, x, y, output_ch*x*y, next_cost, 0, 0)


        if mp_flag==0:
            for k in power_gen(1,3):

                if k*4<x:

                    iD = 'mp' + str(k) + '*' + str(k) + '_' + str(x) + "*" + str(y)+ '_' + str(x/k) + "*" + str(y/k) + '_n_' + str(counter)

                    name = 'Max_Pooling' + str(k) + '*' + str(k) + '_' + str(x) + "*" + str(y)+ '_' + str(x/k) + "*" + str(y/k) + '_n_' + str(counter)

                    next_cost = cost - (((x/k)**2)*sing_comp*prev_dim) - ((x/k)*(y/k)*prev_dim)

                    if next_cost-(prev_dim*out_dim*(x/k)*(y/k)*mul_cost_DN + out_dim)>0:
                        m_tree.create_node(name, iD, parent=c_p, data=[k,k,(x/k),(y/k), prev_dim])
                        print "Pool"
                        tree_creater(iD, prev_dim, x/k, y/k, prev_dim*x*y/(k*k), next_cost, 0, 1)


    for output_ch in power_gen(1,15):

        iD = 'fc' + '_' + str(fc_dim) + "*" + str(output_ch) + '_n_' + str(counter)

        name = 'Dense' + '_' + str(fc_dim) + "*" + str(output_ch) + '_n_' + str(counter)

        next_cost = cost - (fc_dim*output_ch*mul_cost_DN) - (relu_act*output_ch) - (output_ch)

        if next_cost-(output_ch*out_dim*mul_cost_DN + out_dim) > 0:
            m_tree.create_node(name, iD, parent=c_p, data=[fc_dim, output_ch])
            print "Dense"
            tree_creater(iD, output_ch, x, y, output_ch, next_cost, 1, 0)



    counter+=1
    iD = 'out' + '_' + str(fc_dim) + "*" + str(out_dim)+'_n_' + str(counter)
    name = 'Out' + '_' + str(fc_dim) + "*" + str(out_dim)+ '_n_' + str(counter)
    m_tree.create_node(name, iD, parent=c_p, data=[fc_dim,out_dim])



prev_dim=1
tree_creater('r', prev_dim, inp_dimx, inp_dimy, prev_dim*inp_dimx*inp_dimy, cost_br, 0, 1)
m_tree.show()
syntax_generator(m_tree, inp_dimx, inp_dimy)
