# coding:utf8
from datainput import Clothes
from torchvision import transforms as tsfms
from torch.utils.data import DataLoader
import pickle
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
import argparse as agp

parser = agp.ArgumentParser(description='Train resnet')
parser.add_argument('-p', '--pretrain', default=None, help='the root to pretrained model')
parser.add_argument('--prefix', type=str, default='', help='the prefix for saved model param')
parser.add_argument('--olr', type=float, default=0.01, help='the origin learning rate')
args = parser.parse_args()
pretrain = args.pretrain
prefix = args.prefix

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
train_batch = 128 # 64 before  
test_batch = 32
olr = args.olr 
numcls = len(labellst)
is_cuda = torch.cuda.is_available() 
decay_rate = 0.4


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
# load pretrained model
def updatedict(net):
    modelroot = '/home/feijiang/.torch/models/resnet18-5c106cde.pth'
    pretrained_model = torch.load(modelroot)
    net_dict = net.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if not k.startswith('fc')}
    net_dict.update(pretrained_model)
    net.load_state_dict(net_dict)
    return net


net = resnet18()
net.fc = nn.Linear(512, numcls, True)
if pretrain is None:
    net = updatedict(net)
else:
    print(f'load the {pretrain}')
    net.load_state_dict(torch.load(pretrain))
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
        if (idx+1)%1000 == 0:
            torch.save(net.state_dict(), f'./savedoc/net_{epoch+1}_{idx+1}.pkl')
            print('save model', f'./savedoc/{prefix}net_{epoch+1}_{idx+1}.pkl')
            aj_lr(optimizer, decay_rate)
