import struct
import numpy
from PIL import Image, ImageOps

dat_set = '-'

inputs = []
labels = []
inputsTst = []
labelsTst = []
inputs_fin_c = []
labels_c = []
inputsTst_fin_c = []
labelsTst_c = []
inputs_fin_p_shuf = []
labels_p_shuf = []
inputsTst_fin_p = []
labelsTst_p = []

if dat_set == 'MNIST' :
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

if dat_set == 'CIFAR' :
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
        for i in xrange(nsamp_c):
            num = bfile.read(1)
            val = struct.unpack('B', num)[0]
            label = [0 for i in xrange(10)]
            label[val] = 1
            labels_c.append(label)
        print labels_c[0]

        inputs_fin_c=[]
        for i in xrange(nsamp_c):
            inputs_c = []
            nums = bfile.read(inpsize_c*4)
            #print ord(nums)
            inp = list(struct.unpack('3072f', nums))
            inputs_c = [pix_val for pix_val in inp]

            inputs_fin_c.append(inputs_c)
        print len(inputs_fin_c)

    with open("/home/hdft/Documents/Data_ML/2017_07_03 CNN_BIN/CIFAR.test.bin", "rb") as bfile:
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

        labelsTst_c=[]
        for i in xrange(nsamp_c):
            num = bfile.read(1)
            val = struct.unpack('B', num)[0]
            label = [0 for i in xrange(10)]
            label[val] = 1
            labelsTst_c.append(label)
        print labelsTst_c[0]

        inputsTst_fin_c=[]
        for i in xrange(nsamp_c):
            inputs_c = []
            nums = bfile.read(inpsize_c*4)
            #print ord(nums)
            inp = list(struct.unpack('3072f', nums))
            inputs_c = [pix_val for pix_val in inp]

            inputsTst_fin_c.append(inputs_c)

if dat_set == 'PATH':
    with open("/home/hdft/Documents/Data_ML/PATH_DATA.bin", "rb") as bfile:
        #"/home/hdft/Documents/Data_ML/2017_07_03 CNN_BIN/CIFAR.train.bin"
        num = bfile.read(4)
        xdim_p = struct.unpack('i', num)[0]
        num = bfile.read(4)
        ydim_p = struct.unpack('i', num)[0]
        num = bfile.read(4)
        ccod_p = struct.unpack('i', num)[0]
        num = bfile.read(4)
        num = bfile.read(4)
        num = bfile.read(4)
        henc_p = struct.unpack('i', num)[0]
        num = bfile.read(4)
        nsamp_p = struct.unpack('i', num)[0]
        num = bfile.read(4)
        inpsize_p = struct.unpack('i', num)[0]

        print xdim_p, ydim_p, ccod_p, henc_p, nsamp_p, inpsize_p

        labels_p=[]
        for i in xrange(nsamp_p):
            num = bfile.read(1)
            val = struct.unpack('B', num)[0]
            label = [0 for i in xrange(2)]
            label[val] = 1
            labels_p.append(label)
        #print labels_p

        inputs_fin_p=[]
        for i in xrange(nsamp_p):
            inputs_p = []
            nums = bfile.read(inpsize_p*4)
            #print ord(nums)
            inp = list(struct.unpack('12288f', nums))
            inputs_p = [pix_val for pix_val in inp]

            inputs_fin_p.append(inputs_p)
        #print len(inputs_fin_p)

        input_shuffle = [[i,l] for i,l in zip(inputs_fin_p, labels_p)]
        numpy.random.shuffle(input_shuffle)
        numpy.save('/home/hdft/Documents/Data_ML/PATH_Shuffled.npy', input_shuffle)
        # inputs_fin_p = [val[0] for val in input_shuffle]
        # labels_p = [val[1] for val in input_shuffle]
        #print labels_p

def input_inject():
    print "We're good to go"
    return inputs, labels

def test_inject():
    return inputsTst, labelsTst

def input_inject_CIFAR():
    print "We're good to go"
    return inputs_fin_c, labels_c

def test_inject_CIFAR():
    return inputsTst_fin_c, labelsTst_c

inputs_PATH = numpy.load('/home/hdft/Documents/Data_ML/PATH_Shuffled.npy')
inputs_fin_p_shuf = [val[0] for val in inputs_PATH]
labels_p_shuf = [val[1] for val in inputs_PATH]

def input_inject_PATH():
    print "We're good to go"
    return inputs_fin_p_shuf, labels_p_shuf

def test_inject_PATH():
    return inputsTst_fin_p, labelsTst_p

# im = Image.new('RGB', (32,32))
# pixel = im.load()
# count=0
# print pixel[0,0]
# for i in xrange(im.size[0]):
#     for j in xrange(im.size[1]):
#         #val = inputs_fin[1001][count] + inputs_fin[1001][count+1024] + inputs_fin[1001][count+2048]
#         pixel[j,i]=(inputs_fin[4000][count], inputs_fin[4000][count+1024], inputs_fin[4000][count+2048])
#         # pixel[j,i]=tuple(inputs_fin[4000][count])
#         count+=1
#
# print labels_c[4000]
# # ImageOps.invert(im).show()
# im.show()
