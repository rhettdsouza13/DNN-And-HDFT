from PIL import Image
import numpy
import matplotlib.pyplot as pl
import os
import matplotlib
from parser import input_inject_MIT
import matplotlib.gridspec as gridspec

def view1x32(data):                                                             #visulize 1x32 convolve filters
    new_im = Image.new('F', (40,40))
    pastex = 0
    pastey = 0
    for ind in xrange(32):
        im = Image.new('F',(5,5))
        pix = im.load()
        for a in xrange(5):
            for z in xrange(5):
                if data[a][z][0][ind]>0.0:
                    pix[z,a] = 255
                else:
                    pix[z,a] = 0
                # print pix[z,a]

        new_im.paste(im, (pastex, pastey))
        if pastex>=40:
            pastey+=5
            pastex=0
        else:
            pastex+=5
    new_im.show()


def view32x64(data):                                                            #visulize 32x64 convolve filters
    new_im = Image.new('RGB', (380,380))
    pastex = 0
    pastey = 0
    for bInd in xrange(64):
        for ind in xrange(32):
            im = Image.new('RGB',(5,5))
            pix = im.load()
            for a in xrange(5):
                for z in xrange(5):
                    if data[a][z][ind][bInd]>0.0:
                        pix[z,a] = (255,0,0)
                    else:
                        pix[z,a] = (0,0,255)
                # print pix[z,a]

            new_size = (8,8)
            b_im = Image.new("RGB", new_size)
            b_im.paste(im, ((new_size[0]-im.size[0])/2,
                                  (new_size[1]-im.size[1])/2))

            new_im.paste(b_im, (pastex, pastey))
            if pastex>=380:
                pastey+=5
                pastex=0
            else:
                pastex+=5
    new_im.show()
#
# def view32x32Im(data):
#     data = [[numpy.reshape(data, (32,32)), 1]                                         #visualize 32x32 Image Data
#     im = Image.new('RGB', (32,32))
#     pixel = im.load()
#     for i in xrange(im.size[0]):
#         for j in xrange(im.size[1]):
#             if data[0][j][i][0]>0.0:
#                 pixel[i,j]=(255,0,0)
#             else:
#                 pixel[i,j]=(0,0,255)
#     im.show()

def view32x32Im(data):
    data = numpy.reshape(data, (32,32) )                                            #visualize 32x32 Image Data
    im = Image.new('RGB', (32,32))
    pixel = im.load()
    for i in xrange(im.size[0]):
        for j in xrange(im.size[1]):
            if data[j][i]>0.0:
                pixel[i,j]=(255,0,0)
            else:
                pixel[i,j]=(0,0,255)
    im.show()

def view16x16Im(data):                                                          #visualize 16x16 image data
    im = Image.new('F', (16,16))
    pixel = im.load()
    for i in xrange(im.size[0]):
        for j in xrange(im.size[1]):
            if data[0][j][i][0]>0.0:
                pixel[i,j]=255
            else:
                pixel[i,j]=0
    im.show()


def plotter_wv(path, im_list):
    fig, axes = pl.subplots(6,6)
    fig.set_size_inches(2,2)
    matplotlib.rcParams.update({'font.size': 4.7})
    inputs, labels = input_inject_MIT()

    i=0
    row=0
    col=0

    while row!=6 and col!=6:
        # print labels[i]
        if labels[i][1] == 0:
            im = numpy.transpose(numpy.reshape(inputs[i], (3,64,64)),(1,2,0))
            im = ((im + 1) * 127.5).astype(numpy.uint8)
            im = Image.fromarray(im, mode='RGB')
            im = im.resize((250,250))
            im = numpy.asarray(im)
            axes[row,col].imshow(im)
            # axes[0].set_title('Image')
            axes[row,col].get_xaxis().set_visible(False)
            axes[row,col].get_yaxis().set_visible(False)
            print row, col
            if col == 5:
                row += 1
                col=0
            else:
                col += 1
        i+=1

    # print im_list
    # for i,ims in enumerate(im_list):
    #     im = Image.open(path + ims)
    #     im = im.resize((350,350)).convert(mode='L')
    #     im = numpy.asarray(im)
    #     axes[i+1].imshow(im, cmap='gray')
    #     axes[i+1].set_title('Convolutional Layer ' + str(i+1))
    #     axes[i+1].get_xaxis().set_visible(False)
    #     axes[i+1].get_yaxis().set_visible(False)

    fig.savefig(base_path + 'together0.png', dpi=300, format='png', bbox_inches='tight')
    png2 = Image.open(base_path + 'together0.png')
    png2.save(base_path + 'together0.tiff', compression='lzw')
    pl.show()

base_path = '/home/hdft/Documents/DNN-Complete/DNN-PLOTS/Box_Plots/'
# plotter_wv(base_path, sorted(os.listdir(base_path)))
#

def paster(base_path):
    fig,axes = pl.subplots(1,2)
    fig.set_size_inches(4,2)
    matplotlib.rcParams.update({'font.size': 3.0})

    im = Image.open(base_path + "together.png")
    # im = im.resize((1200,1200))
    im = numpy.asarray(im)
    axes[0].imshow(im)
    # axes[0].set_title('Positive-labeled Examples\n(Cells Undergoing Mitosis)')
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].axis('off')

    im2 = Image.open(base_path + "together0.png")
    # im2 = im2.resize((1200,600))
    im2 = numpy.asarray(im2)
    print im2.shape
    axes[1].imshow(im2)
    # axes[1].set_title('Negative-labeled Examples\n(Cells Not Undergoing Mitosis)')
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[1].axis('off')

    pl.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(base_path + 'all_tog.png', dpi=300, format='png', bbox_inches='tight')
    png2 = Image.open(base_path + 'all_tog.png')
    png2.save(base_path + 'all_tog.tiff', compression='lzw')

    # pl.show()



paster(base_path)


# def image_resizer(lin_im, k):
#     lin_im_np = numpy.array(lin_im)
#     lin_im_2d = numpy.reshape(lin_im_np,(32,32))
