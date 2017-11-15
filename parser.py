import struct
import numpy
from PIL import Image



with open("/home/hdft/Documents/Data_ML/2017_07_03 CNN_BIN/MINST.train.bin", "rb") as bfile:
    num = bfile.read(4)
    xdim = struct.unpack('i', num)[0]
    num = bfile.read(4)
    ydim = struct.unpack('i', num)[0]
    num = bfile.read(4)
    ccod = struct.unpack('i', num)[0]
    num = bfile.read(4)
    num = bfile.read(4)
    num = bfile.read(4)
    henc = struct.unpack('i', num)[0]
    num = bfile.read(4)
    nsamp = struct.unpack('i', num)[0]
    num = bfile.read(4)
    inpsize = struct.unpack('i', num)[0]

    labels=[]
    for i in xrange(nsamp):
        num = bfile.read(1)
        val = struct.unpack('B', num)[0]
        label = [0 for i in xrange(10)]
        label[val] = 1
        labels.append(label)

    inputs=[]
    for i in xrange(nsamp):
        nums = bfile.read(inpsize*4)
        inp = list(struct.unpack('1024f', nums))
        inputs.append(inp)

with open("/home/hdft/Documents/Data_ML/2017_07_03 CNN_BIN/MINST.test.bin", "rb") as bfile:
    num = bfile.read(4)
    xdim = struct.unpack('i', num)[0]
    num = bfile.read(4)
    ydim = struct.unpack('i', num)[0]
    num = bfile.read(4)
    ccod = struct.unpack('i', num)[0]
    num = bfile.read(4)
    num = bfile.read(4)
    num = bfile.read(4)
    henc = struct.unpack('i', num)[0]
    num = bfile.read(4)
    nsamp = struct.unpack('i', num)[0]
    num = bfile.read(4)
    inpsize = struct.unpack('i', num)[0]

    labelsTst=[]
    for i in xrange(nsamp):
        num = bfile.read(1)
        val = struct.unpack('B', num)[0]
        label = [0 for i in xrange(10)]
        label[val] = 1
        labelsTst.append(label)

    inputsTst=[]
    for i in xrange(nsamp):

        nums = bfile.read(inpsize*4)

        inp = list(struct.unpack('1024f', nums))
        inputsTst.append(inp)


def input_inject():
    print "We're good to go"
    return inputs, labels

def test_inject():
    return inputsTst, labelsTst
# im = Image.new('I', (32,32))
# pixel = im.load()
# count=0
# for i in xrange(im.size[0]):
#     for j in xrange(im.size[1]):
#         pixel[j,i]=inp[count]
#         count+=1
# print labels[10]
# im.show()
