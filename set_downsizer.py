from parser import *
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from visualizer import *
from random import randrange
import numpy
#
inputs_t, labels_t = input_inject_CIFAR()

def downsize_me(inputs,labels):
    comp_coll = [[] for i in xrange(10)]
    out_nums = []
    #view32x32Im(inputs[0])

    iters = 0
    for inp,label in zip(inputs, labels):
        #print label.index(1),
        comp_coll[label.index(1)].append(inp)

    # view32x32Im(comp_coll[3][4])

    for num in comp_coll:
        print len(num)
    iters = 0
    for num in comp_coll:
        curr_num = num
        var_len = len(curr_num)
        while var_len > 1000:
            next_num = []
            if var_len%2 == 1:
                next_num.append(curr_num.pop(randrange(len(curr_num)-1)))

            for i in xrange(var_len/2):
                # print var_len
                x1 = curr_num.pop(randrange(len(curr_num)-1))
                if len(curr_num) > 1 :
                    x2 = curr_num.pop(randrange(len(curr_num)-1))
                else:
                    x2 = curr_num.pop(0)

                corr = pearsonr(x1,x2)
                if corr > 0.98:
                    next_num.append(x1)
                else:
                    next_num.append(x1)
                    next_num.append(x2)
            curr_num = []
            curr_num = next_num
            var_len = len(curr_num)
        label_o = [0 for i in xrange(10)]
        label_o[iters] = 1
        for inp in next_num:
            out_nums.append([inp, label_o])

        iters+=1

    # print out_nums[0]
    numpy.random.shuffle(out_nums)
    numpy.array(out_nums)

    return [inp[0] for inp in out_nums], [inp[1] for inp in out_nums]

# ins, outs = downsize_me(inputs_t, labels_t)
# print ins[0], outs[0]
def downsize_me_kmeans(inputs,labels):
    comp_coll = [[] for i in xrange(10)]
    out_nums = []
    #view32x32Im(inputs[0])
    sub_set_size = 100
    iters = 0
    for inp,label in zip(inputs, labels):
        #print label.index(1),
        comp_coll[label.index(1)].append(inp)

    # view32x32Im(comp_coll[3][4])

    for num in comp_coll:
        print len(num)
    iters = 0
    for num in comp_coll:
        data_dic = {}
        estimator = KMeans(n_clusters=sub_set_size)
        estimator.fit(num)
        data_dic = {i: numpy.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
        print len(data_dic[0])
        print estimator.labels_
        class_fin = []
        print data_dic
        print "Class done"
        for i in xrange(sub_set_size):
            class_rep = num[data_dic[i][randrange(len(data_dic[i]))]]
            class_fin.append(class_rep)
        # print class_fin[0]
        label_o = [0 for i in xrange(10)]
        label_o[iters] = 1
        for sample in class_fin:
            out_nums.append([sample, label_o])
    iters += 1
    print out_nums[0]
    numpy.random.shuffle(out_nums)
    numpy.array(out_nums)
    numpy.save("downsized_CIFAR.npy", out_nums)
    return [inp[0] for inp in out_nums], [inp[1] for inp in out_nums]

ins, outs = downsize_me_kmeans(inputs_t, labels_t)
print ins[0], outs[0]
