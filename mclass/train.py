# coding:utf8
from datainput import MLClothes
from torchvision import transforms as tsfms
from torch.utils.data import DataLoader
import pickle
from torchvision.models import resnet18, resnet50, resnet101
import torch
import torch.nn as nn
import torch.optim as optim
import argparse as agp

parser = agp.ArgumentParser(description='Train multilabels resnet')
parser.add_argument('-p', '--pretrain', default=None, help='the root to pretrained model')
parser.add_argument('--prefix', type=str, default='ml', help='the prefix for saved model param')
parser.add_argument('--olr', type=float, default=0.001, help='the origin learning rate')
parser.add_argument('--model', type=str, default='resnet18', help='the used model', choices=['resnet18', 'resnet50', 'resnet101'])
args = parser.parse_args()
pretrain = args.pretrain
prefix = args.prefix
model = args.model

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
    return torch.FloatTensor(label) 

# training parameters
btdic = {'resnet18':128, 'resnet50':64, 'resnet101':32}
epochs = 60
train_batch = btdic[model]# 64 before  
test_batch = 32
olr = args.olr 
numcls = len(labellst)
is_cuda = torch.cuda.is_available() 
decay_rate = 0.2
decay_num = [int(epochs*0.6), int(epochs*0.2), int(epochs*0.1)]
mfc= {}
mfc['resnet18'] = 512
mfc['resnet50'] = 2048 
mfc['resnet101'] = 2048 

trainroot = './savedoc/mlabelstrain.pkl' 
testroot = './savedoc/mlabelstest.pkl' 
valroot = './savedoc/mlabelsval.pkl' 


# use mean and std of imagenet. Something is wrong with the true mean and std
tsfm = tsfms.Compose([tsfms.Resize([224, 224]), tsfms.ToTensor(), tsfms.Normalize(mean=imagenetmean, std=imagenetstd)])
traincls = MLClothes(trainroot, tsfm, ttsfm)
testcls = MLClothes(testroot, tsfm, ttsfm)
traindata = DataLoader(dataset=traincls, batch_size=train_batch, shuffle=True)
testdata = DataLoader(dataset=testcls, batch_size=test_batch, shuffle=True)

# model
# load pretrained model
def updatedict(net, model):
    if model == 'resnet18':
        modelroot = '/home/feijiang/.torch/models/resnet18-5c106cde.pth'
    elif model == 'resnet50':
        modelroot = '/home/feijiang/.torch/models/resnet50-19c8e357.pth'
    elif model == 'resnet101':
        modelroot = '/home/feijiang/.torch/models/resnet101-5d3b4d8f.pth'
    pretrained_model = torch.load(modelroot)
    net_dict = net.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if not k.startswith('fc')}
    net_dict.update(pretrained_model)
    net.load_state_dict(net_dict)
    return net

if model == 'resnet18':
    net = resnet18()
elif model == 'resnet50':
    net = resnet50()
elif model == 'resnet101':
    net = resnet101()
net.fc = nn.Linear(mfc[model], numcls, True)
sf = nn.Softmax(dim=1)
if pretrain is None:
    net = updatedict(net, model)
else:
    print(f'load the {pretrain}')
    net.load_state_dict(torch.load(pretrain))
if is_cuda: net = net.cuda()
#criterion = nn.MultiLabelSoftMarginLoss()
#criterion = nn.BCELoss(reduction='elementwise_mean')
optimizer = optim.Adam(net.parameters(), lr=olr, weight_decay=0)

# training
for epoch in range(1, epochs+1):
    run_loss = 0.0
    for idx, data in enumerate(traindata, 1):
        imgs, labels = data
        if is_cuda: imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        net.train()
        weights= labels.clone()
        weights[weights==0]=0.1
        criterion = nn.BCELoss(weight=weights, reduction='elementwise_mean')
        outputs = net(imgs)
        outputs = sf(outputs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        if idx % 10 == 0:
            print("epoch:", epoch, "iters:",  idx, run_loss/10)
            run_loss = 0.0
    torch.save(net.state_dict(), f'./savedoc/{prefix}net_{epoch}.pkl')
    print('save model', f'./savedoc/{prefix}net_{epoch}.pkl')
    if epoch in decay_num:
        aj_lr(optimizer, decay_rate)

