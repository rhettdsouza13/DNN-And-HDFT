from PIL import Image
import numpy

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

def view32x32Im(data):                                                          #visualize 32x32 Image Data
    im = Image.new('RGB', (32,32))
    pixel = im.load()
    for i in xrange(im.size[0]):
        for j in xrange(im.size[1]):
            if data[0][j][i][0]>0.0:
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
