# coding:utf8
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms as tsfms
from PIL import Image

# get the labels
with open('./documents/labels.txt') as f:
    labels = f.readlines()
labels = [i.strip() for i in labels]

# mean, std of imagenet images
imagenetmean = [0.485, 0.456, 0.406]
imagenetstd = [0.229, 0.224, 0.225]

# the function to make image into tensor and normalize it
tsfm = tsfms.Compose([tsfms.Resize([224, 224]), tsfms.ToTensor(), tsfms.Normalize(mean=imagenetmean, std=imagenetstd)])

# some parameters
is_cuda = torch.cuda.is_available()

# get the net
net = resnet18(True)
net.eval()
if is_cuda:
    net = net.cuda()

# get the image
im = Image.open('./documents/1.jpg')

# run the net
imts = tsfm(im)
imts = imts.unsqueeze(0)
if is_cuda:
    imts = imts.cuda()
out = net(imts)
outprob = nn.Softmax(dim=1)(out)
cls = out.cpu().detach().numpy().argmax()
print(labels[cls], cls)
