# coding:utf8
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101
from torchvision import transforms as tsfms
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFile
from pathlib import Path
import numpy as np
from torchvision.models import resnet18
import pickle
from datainput import MLClothes
import argparse as agp

parser = agp.ArgumentParser(description='test the multiclasses model')
parser.add_argument('--root', type=str, default='./savedoc/mlabelsval.pkl', help='the root to images')
parser.add_argument('--net', type=str,  help='The saved model to evaluate')
parser.add_argument('--num', type=int, default=0, help='num of batches to evaluate, when num is 0, evaluate the whole datasets')
parser.add_argument('--batchsize', '-b', type=int, default=8, help='num of images per batches to test')
parser.add_argument('--is_exact', action='store_true',  help='specify which measurement to use')
parser.add_argument('--model', type=str, default='resnet18', help='the used model', choices=['resnet18', 'resnet50', 'resnet101'])
args = parser.parse_args()

root = Path(args.root)
netmd = args.net
num2test = args.num
is_exact = args.is_exact
model = args.model

def diff(x):
    x1 = np.concatenate([x[1:], [0]])
    return (x-x1)[:-1]


def ink(gt, pre, k=5):
    returnlst = []
    for gt1, pre1 in zip(gt, pre):
        gtlabel = np.argwhere(gt1==1).reshape(-1)
        sortidx = np.argsort(-pre1)[:k]
        returnlst.append(set(gtlabel)<=set(sortidx))
    return returnlst

def exactt(gt, pre):
    returnlst = []
    for gt1, pre1, in zip(gt, pre):
        sortidx = np.argsort(-pre1)
        sortpre1 = pre1[sortidx]
        difk = diff(sortpre1)[:5]
        num = np.argmax(difk[1:])+2
        gtlabel = np.argwhere(gt1==1).reshape(-1)
        returnlst.append(set(gtlabel)==set(sortidx[:num]))
    return returnlst


def maxk(arr, k):
    argidx = np.argsort(-arr)
    return argidx[:, :k]



def evalt(imlist, net, tsfm):
    imlist = [tsfm(im) for im in imlist]
    inputs = torch.stack(imlist)
    if is_cuda: inputs = inputs.cuda()
    outputs = net(inputs)
    return outputs


ImageFile.LOAD_TRUNCATED_IMAGES=True

with open('./savedoc/labels.pkl','rb') as f:
    labellst = pickle.load(f)


def ttsfm(label):
    return torch.FloatTensor(label)


# mean, std of imagenet images
imagenetmean = [0.485, 0.456, 0.406]
imagenetstd = [0.229, 0.224, 0.225]


# the function to make image into tensor and normalize it
tsfm = tsfms.Compose([tsfms.Resize([224, 224]), tsfms.ToTensor(), tsfms.Normalize(mean=imagenetmean, std=imagenetstd)])


# some parameters
#valroot = '/home/feijiang/datasets/imgval'
valroot = root
is_cuda = torch.cuda.is_available()
numcls = len(labellst)
val_batch = args.batchsize 
mfc = {'resnet18':512, 'resnet50':2048, 'resnet101':2048}


valcls = MLClothes(valroot, tsfm, ttsfm)
valdata = DataLoader(dataset=valcls, batch_size=val_batch, shuffle=True)


# test
if model == 'resnet18':
    net = resnet18()
elif model == 'resnet50':
    net = resnet50()
elif model == 'resnet101':
    net = resnet101()
net.fc = nn.Linear(mfc[model], numcls, True)
net.load_state_dict(torch.load(netmd))
if is_cuda: net = net.cuda()
net.eval()

reslst = []
#saved_file = []
for idx, data in enumerate(valdata):
    imgs, labels = data
    if is_cuda: imgs, labels = imgs.cuda(), labels.cuda()
    outputs = net(imgs)
    npout = outputs.cpu().detach().numpy()
    nplabel = labels.cpu().numpy()
#    saved_file.append([npout, nplabel])
    if is_exact:
        reslst += exactt(nplabel, npout)
    else:
        reslst += ink(nplabel, npout)
    print(f'num of images, {len(reslst)}', f'{100*(np.array(reslst)>0).sum()/len(reslst):.3f}%') 
    if (idx + 1) == num2test:
        break

