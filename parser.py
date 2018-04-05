import struct
import numpy
from PIL import Image, ImageOps

dat_set = 'MIT2'

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
inputs_fin_m = []
labels_m = []

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
if dat_set == 'PATH2':
    inputs_PATH = numpy.load('/home/hdft/Documents/Data_ML/PATH_Shuffled.npy')
    inputs_fin_p_shuf = [val[0] for val in inputs_PATH]
    labels_p_shuf = [val[1] for val in inputs_PATH]


if dat_set == 'MITOSIS':
    with open("/home/hdft/Documents/Data_ML/mitosis.bin", "rb") as bfile:
        #"/home/hdft/Documents/Data_ML/2017_07_03 CNN_BIN/CIFAR.train.bin"
        num = bfile.read(4)
        xdim_m = struct.unpack('i', num)[0]
        num = bfile.read(4)
        ydim_m = struct.unpack('i', num)[0]
        num = bfile.read(4)
        ccod_m = struct.unpack('i', num)[0]
        num = bfile.read(4)
        num = bfile.read(4)
        num = bfile.read(4)
        henc_m = struct.unpack('i', num)[0]
        num = bfile.read(4)
        nsamp_m = struct.unpack('i', num)[0]
        num = bfile.read(4)
        inpsize_m = struct.unpack('i', num)[0]

        print xdim_m, ydim_m, ccod_m, henc_m, nsamp_m, inpsize_m

        labels_m=[]
        for i in xrange(nsamp_m):
            num = bfile.read(1)
            val = struct.unpack('B', num)[0]
            label = [0 for i in xrange(2)]
            label[val] = 1
            labels_m.append(label)


        inputs_fin_m=[]
        for i in xrange(nsamp_m):
            inputs_m = []
            nums = bfile.read(inpsize_m*4)
            #print ord(nums)
            inp = list(struct.unpack('12288f', nums))
            inputs_m = [pix_val for pix_val in inp]

            inputs_fin_m.append(inputs_m)

        count_1 = 0
        count_0 = 0
        new_count_0 = 0
        new_count_1 = 0

        inputs_shuff_p = []
        print labels_m[7201][0]
        print len(inputs_fin_m)

        for i in xrange(nsamp_m):
            if labels_m[i][0] == 1:
                count_0 += 1
                if count_0 <= 2860:
                    inputs_shuff_p.append([inputs_fin_m[i],labels_m[i]])
                    # labels_config.append(labels_m[i])
                    # inputs_fin_m_config.append(inputs_fin_m[i])
            elif labels_m[i][1] == 1:
                count_1 += 1
                inputs_shuff_p.append([inputs_fin_m[i],labels_m[i]])
                # labels_config.append(labels_m[i])
                # inputs_fin_m_config.append(inputs_fin_m[i])

        # print len(labels_config)
        # print len(inputs_fin_m_config)
        #
        # for i in xrange(len(inputs_fin_m_config)):
        #     if labels_config[i][0] == 1:
        #         new_count_0 += 1
        #     elif labels_config[i][1] == 1:
        #         new_count_1 += 1

        numpy.random.shuffle(inputs_shuff_p)
        numpy.save('/home/hdft/Documents/Data_ML/MIT_Shuffled.npy', inputs_shuff_p)
        print count_1, count_0
        print new_count_1, new_count_0

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



def input_inject_PATH():
    print "We're good to go"
    return inputs_fin_p_shuf, labels_p_shuf

def test_inject_PATH():
    return inputsTst_fin_p, labelsTst_p

if dat_set == 'MIT2':
    inputs_PATH = numpy.load('/home/hdft/Documents/Data_ML/MIT_Shuffled.npy')
    inputs_fin_m_shuf = [val[0] for val in inputs_PATH]
    labels_m_shuf = [val[1] for val in inputs_PATH]

def input_inject_MIT():
    print "We're good to go"
    return inputs_fin_m_shuf, labels_m_shuf

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
