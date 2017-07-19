from treelib import *

def syntax_generator(tree):
    paths = tree.paths_to_leaves()

    for path in paths:

        net = '32,32,1|'

        for node in path:

            if 'fc' in node:
                dim = tree.get_node(node).data
                net+='full,tanh|'
                net+='1,1,'+str(dim[1]) + '|'
                continue
            if 'conv' in node:
                dim = tree.get_node(node).data
                net+='conv,tanh,' + str(dim[0]) + '|' + str(dim[4]) + ',' + str(dim[5]) + ',' + str(dim[3]) + '|'
                continue
            if 'mp' in node:
                dim = tree.get_node(node).data
                net+='max_pooling,identity,' + str(dim[0]) + '|' + str(dim[2]) + ',' + str(dim[3]) + ',' + str(dim[4]) + '|'
                continue
            if 'out' in node:
                dim = tree.get_node(node).data
                net+='full,tanh|'
                net+='1,1,'+str(dim[1])
                continue
        print ""
        print net
