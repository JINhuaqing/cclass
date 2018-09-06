# coding:utf8

from datainput import Clothes
from torchvision import transforms as tsfms
from torch.utils.data import DataLoader
import pickle
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim

def aj_lr(optim, decay_rate=0.1):
    for pg in optim.param_groups:
        pg['lr'] = pg['lr'] * decay_rate

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
train_batch = 256 # 64 before  
test_batch = 32
olr = 0.1
numcls = len(labellst)
is_cuda = torch.cuda.is_available() 


trainroot = '/home/feijiang/datasets/imgtrain'
testroot = '/home/feijiang/datasets/imgtest'
valroot = '/home/feijiang/datasets/imgval'

# use mean and std of imagenet. Something is wrong with the true mean and std
tsfm = tsfms.Compose([tsfms.Resize([224, 224]), tsfms.ToTensor(), tsfms.Normalize(mean=imagenetmean, std=imagenetstd)])
traincls = Clothes(trainroot, tsfm, ttsfm)
testcls = Clothes(testroot, tsfm, ttsfm)
traindata = DataLoader(dataset=traincls, batch_size=train_batch, shuffle=True)
testdata = DataLoader(dataset=testcls, batch_size=test_batch, shuffle=True)


# model
net = resnet18()
net.fc = nn.Linear(512, numcls, True)
if is_cuda: net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=olr, momentum=0.9)

# training
for epoch in range(epochs):
    run_loss = 0.0
    for idx, data in enumerate(traindata):
        imgs, labels = data
        if is_cuda: imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        net.train()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        if idx % 10 == 9:
            print("epoch:", epoch+1, "iters:",  idx+1, run_loss/10)
            run_loss = 0.0
    aj_lr(optimizer, 0.8)
    torch.save(net.state_dict(), f'./savedoc/net_{epoch}.pkl')
