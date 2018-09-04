# coding:utf8

from datainput import Clothes
from torchvision import transforms as tsfms
from torch.utils.data import DataLoader
import pickle
from torchvision.models import resnet18
import torch

with open('./savedoc/mean.pkl', 'rb') as fmean:
    mean = pickle.load(fmean)
with open('./savedoc/std.pkl', 'rb') as fstd:
    std = pickle.load(fstd)
mean, std = mean/256, std/256
imagenetmean = [0.485, 0.456, 0.406]
imagenetstd = [0.229, 0.224, 0.225]
with open('./savedoc/labels.pkl','rb') as f:
    labellst = pickle.load(f)

def ttsfm(label):
    return labellst.index(label)

# training parameters
epochs = 10
train_batch = 32
test_batch = 32
olr = 0.1
is_cuda = torch.cuda.is_available() 


root = '/home/feijiang/datasets/images'
# use mean and std of imagenet. Something is wrong with the true mean and std
tsfm = tsfms.Compose([tsfms.Resize([224, 224]), tsfms.ToTensor(), tsfms.Normalize(mean=imagenetmean, std=imagenetstd)])
cls = Clothes(root, tsfm, ttsfm)
cls_data = DataLoader(dataset=cls, batch_size=4, shuffle=True)


# model
net = resent18()
if is_cuda: net = net.cuda()

# training
for epoch in range(epochs):
    pass
