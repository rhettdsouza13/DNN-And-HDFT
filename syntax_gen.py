from treelib import *

def syntax_generator(tree, x, y):
    paths = tree.paths_to_leaves()

    with open('nets_list3.txt', 'w') as netfile:

        for path in paths:

            net = str(x) + ',' + str(y) + ',' + str(1) + '|'

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
                    continue

            print net

            netfile.write(net + '\n')
    print len(paths)
