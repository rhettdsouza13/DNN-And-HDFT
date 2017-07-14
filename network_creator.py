from treelib import *


cost_br = 512
inp_dim = 1024
inp_dimx = 32
inp_dimy = 32
out_dim = 10
sing_comp = 1./10
relu_act = 1./100
tanh_act = 1./50

m_tree = Tree()

def power_gen(start,end):
    l_2_exp = [2**x for x in xrange(start,end)]
    return l_2_exp


m_tree.create_node("Root", 'r')
counter = 0

def tree_creater(parent,prev_dim,cost):
    global counter
    counter+=1
    if cost<=0:
        return
    c_p=parent
    for k in power_gen(1,4):

        for output_ch in power_gen(3,10):

            iD = 'c' + str(k) + '*' + str(k) + '_' + str(prev_dim) + "*" + str(output_ch) + '_n_' + str(counter)

            name = 'Convolution' + str(k) + '*' + str(k) + '_' + str(prev_dim) + "*" + str(output_ch) + '_n_' + str(counter)
            m_tree.create_node(name, iD, parent=c_p, data=[k,k,prev_dim,output_ch])

            next_cost = cost - (k*k*prev_dim*output_ch)

            tree_creater(iD,output_ch,next_cost)

prev_dim=1
tree_creater('r',prev_dim, cost_br)
m_tree.show()
