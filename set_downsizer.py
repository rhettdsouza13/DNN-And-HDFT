from parser import *
from scipy.stats import pearsonr
from visualizer import *
from random import randrange

inputs, labels = input_inject()
comp_coll = [[] for i in xrange(10)]

#view32x32Im(inputs[0])

iters = 0
for inp,label in zip(inputs, labels):
    #print label.index(1),
    comp_coll[label.index(1)].append(inp)

# view32x32Im(comp_coll[3][4])

for num in comp_coll:
    print len(num)

for num in comp_coll:
    while len(num) > 50:
        curr_num = num
        next_num = []
        if len(num)%2 == 1:
            curr_num.append(num.pop(randrange(len(num)-1)))
        x1 = curr_num.pop(randrange(len(num)-1))
        x2 = curr_num.pop(randrange(len(num)-1))
        corr = pearsonr(x1,x2)
        if corr > 0.7:
            next_num.append(x1)
        else:
            curr_num.append(x1)
            curr_num.append(x2)
