import os

direc = "/home/hdft/Documents/DNN-Data-Run-203-40-CIFAR-4000-1000/"
for filename in os.listdir(direc):
    parts = filename.split('_')
    num = parts[-1] + ".npy"
    parts[-1] = str(num)
    new_file = '_'.join(parts)
    os.rename(direc + filename, direc + new_file)
