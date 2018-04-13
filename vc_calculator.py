import numpy
import math
#32,32,1|max_pooling,identity,4|8,8,1|conv,relu,5|8,8,20|full,relu|1,1,10


def calc_vc(network):
    net_parts = network.split('|')
    weights = 0
    prev_dim = 1
    out_dim = 0
    full_dim = 0
    layer_num = 0
    conv_fl = 0
    full_fl = 0
    mp_fl = 0
    k = 0
    for part in net_parts:



        if 'max_pooling' in part:
            # print "Max_pooling"
            # print
            continue
        elif 'conv' in part:
            # print "Caught Conv"
            layer_num += 1
            # print
            k = int(part.split(',')[2])
            conv_fl = 1
            continue
        elif 'full' in part:
            # print "Caught Full"
            layer_num += 1
            # print
            full_fl = 1
            continue
        elif conv_fl == 0 and full_fl == 0:
            # print "Storing Init Or Max Dims"
            # print
            prev_dim = int(part.split(',')[2])
            full_dim = int(part.split(',')[0])*int(part.split(',')[1])*int(part.split(',')[2])
            # print prev_dim
            # print full_dim
            # print
            continue
        elif conv_fl:
            # print "Conv weight"
            out_dim = int(part.split(',')[2])
            conv_fl = 0
            weights += prev_dim*out_dim*k*k + out_dim
            prev_dim = int(part.split(',')[2])
            full_dim = int(part.split(',')[0])*int(part.split(',')[1])*int(part.split(',')[2])
            # print weights
            # print
        elif full_fl:
            # print "Full weight"
            out_dim = int(part.split(',')[2])
            full_fl = 0
            weights += full_dim*out_dim + out_dim
            prev_dim = int(part.split(',')[2])
            full_dim = int(part.split(',')[0])*int(part.split(',')[1])*int(part.split(',')[2])
            # print weights
            # print

    vc_dim = weights*(layer_num)*numpy.log2(weights)
    # print layer_num
    # print vc_dim
    return vc_dim

#print calc_vc("32,32,1|conv,relu,5|32,32,20|max_pooling,identity,2|16,16,20|conv,relu,5|16,16,20|conv,relu,7|16,16,20|max_pooling,identity,2|8,8,20|conv,relu,5|8,8,20|conv,relu,5|8,8,20|conv,relu,5|8,8,20|conv,relu,5|8,8,20|full,relu|1,1,10")


def binner(X,Y,N):
    ranger = max(X) - min(X)
    #print min(X)
    bins = numpy.array([i for i in range(int(math.floor(min(X))), int(math.ceil(max(X))), int(math.ceil(ranger/N)))])
    bins = numpy.append(bins, [int(math.ceil(max(X)))])
    #print bins
    inds = numpy.digitize(X, bins)
    binned_Y = [[] for i in xrange(len(bins))]
    #print max(inds)
    for i in xrange(len(Y)):
        binned_Y[inds[i]-1].append(Y[i])
    # for i in binned_Y:
    #     print i
    #     print "Next"
    # for i in inds:
    #     if i == 20:
    #         print i
    #print binned_Y[18]
    return binned_Y, bins
