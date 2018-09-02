# coding:utf8
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms as tsfms
from PIL import Image
from pathlib import Path
import numpy as np


def evalt(imlist, net, tsfm):
    imlist = [tsfm(im) for im in imlist]
    inputs = torch.stack(imlist)
    if is_cuda: inputs = inputs.cuda()
    outputs = net(inputs)
    return outputs


# get the labels
with open('./documents/labels.txt') as f:
    labels = f.readlines()
labels = [i.strip() for i in labels]
labels = np.array(labels)

# mean, std of imagenet images
imagenetmean = [0.485, 0.456, 0.406]
imagenetstd = [0.229, 0.224, 0.225]

# the function to make image into tensor and normalize it
tsfm = tsfms.Compose([tsfms.Resize([224, 224]), tsfms.ToTensor(), tsfms.Normalize(mean=imagenetmean, std=imagenetstd)])

# some parameters
is_cuda = torch.cuda.is_available()
numIms = 10
root = Path('/home/feijiang/datasets/images')

# get the net
net = resnet18(True)
net.eval()
if is_cuda:
    net = net.cuda()

# run the net
imnamelist = []
imlist = []
flag = 1
f = open('./savedoc/test.txt', 'w')
for p1 in root.iterdir():
    for p2 in p1.iterdir():
        im = Image.open(p2)
        if len(im.getbands()) == 3:
            imlist.append(im)
            imnamelist.append(str(p2))
        if len(imlist) == numIms:
            outs = evalt(imlist, net, tsfm)
            cls = outs.cpu().detach().numpy().argmax(axis=1)
            print(flag, *(labels[cls]), sep='---')
            fileout = [ imnamelist[i]+' '+labels[cls][i]+'\n' for i in range(len(imlist))]
            f.writelines(fileout)
            imlist = []
            imnamelist = []
            flag += 1
f.close()
