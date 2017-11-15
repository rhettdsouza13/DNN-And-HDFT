from parser import *
from scipy.stats import pearsonr
from visualizer import *

inputs, labels = input_inject()
comp_coll = [[] for i in xrange(10)]

#view32x32Im(inputs[0])

iters = 0
for inp,label in zip(inputs, labels):
    #print label.index(1),
    comp_coll[label.index(1)].append(inp)

view32x32Im(comp_coll[3][4])
