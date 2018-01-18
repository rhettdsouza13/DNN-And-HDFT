import struct
import numpy
from PIL import Image, ImageOps



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


with open("/home/hdft/Documents/Data_ML/2017_07_03 CNN_BIN/CIFAR.train.bin", "rb") as bfile:
    #"/home/hdft/Documents/Data_ML/2017_07_03 CNN_BIN/CIFAR.train.bin"
    num = bfile.read(4)
    xdim_c = struct.unpack('i', num)[0]
    num = bfile.read(4)
    ydim_c = struct.unpack('i', num)[0]
    num = bfile.read(4)
    ccod_c = struct.unpack('i', num)[0]
    num = bfile.read(4)
    num = bfile.read(4)
    num = bfile.read(4)
    henc_c = struct.unpack('i', num)[0]
    num = bfile.read(4)
    nsamp_c = struct.unpack('i', num)[0]
    num = bfile.read(4)
    inpsize_c = struct.unpack('i', num)[0]

    print xdim_c, ydim_c, ccod_c, henc_c, nsamp_c, inpsize_c

    labels_c=[]
    for i in xrange(nsamp):
        num = bfile.read(1)
        val = struct.unpack('B', num)[0]
        label = [0 for i in xrange(10)]
        label[val] = 1
        labels_c.append(label)
    print labels_c[0]

    inputs_fin=[]
    for i in xrange(nsamp):
        inputs_c = []
        nums = bfile.read(inpsize_c*4)
        #print ord(nums)
        inp = list(struct.unpack('3072f', nums))
        #
        # # inputs_c = [int(pix_val*255.0) for pix_val in inp]
        inputs_c = [ 0 if pix_val < 0.0 else 255 for pix_val in inp]
        # inputs_c = [pix_val for pix_val in inp]

        inputs_fin.append(inputs_c)
        # red = []
        # gr =[]
        # bl = []
        # next_inp = []
        # counter = 0
        # while counter < inpsize_c-2:
        #     # red.append(inputs_c[counter])
        #     # gr.append(inputs_c[counter+1024])
        #     # bl.append(inputs_c[counter+2048])
        #     next_inp.append([inputs_c[counter], inputs_c[counter+1], inputs_c[counter+2]])
        #     counter += 3
        #
        # inputs_fin.append(next_inp)

    print inputs_fin[4000]

def input_inject():
    print "We're good to go"
    return inputs, labels

def test_inject():
    return inputsTst, labelsTst


im = Image.new('RGB', (32,32))
pixel = im.load()
count=0
print pixel[0,0]
for i in xrange(im.size[0]):
    for j in xrange(im.size[1]):
        #val = inputs_fin[1001][count] + inputs_fin[1001][count+1024] + inputs_fin[1001][count+2048]
        pixel[j,i]=(inputs_fin[4000][count], inputs_fin[4000][count+1024], inputs_fin[4000][count+2048])
        # pixel[j,i]=tuple(inputs_fin[4000][count])
        count+=1

print labels_c[4000]
# ImageOps.invert(im).show()
im.show()
