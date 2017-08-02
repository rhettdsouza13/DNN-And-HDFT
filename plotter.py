import matplotlib.pyplot as pl
import os
import numpy

data_dir = '/home/hdft/Documents/DNN-Data/'

for data_file in os.listdir(data_dir):

    data = numpy.load(data_dir+data_file)
    test_accuracy= data[1:,0]
    test_error= data[1:,1]
    train_accuracy= data[1:,2]
    train_error = data[1:,3]

    pl.plot(range(len(test_error)),test_error)
    pl.plot(range(len(train_error)),train_error)
	
    
pl.show()
