# coding:utf8
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms as tsfms
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFile
from pathlib import Path
import numpy as np
from torchvision.models import resnet18
import pickle
from datainput import testClothes
import argparse as agp

parser = agp.ArgumentParser(description='test the model')
parser.add_argument('--root', type=str, default='/home/feijiang/datasets/imgval', help='the root to images')
parser.add_argument('--net', type=str,  help='The saved model to evaluate')
parser.add_argument('--num', type=int, default=0, help='num of epochs to evaluate, when num is 0, evaluate the whole datasets')
args = parser.parse_args()

root = Path(args.root)
netmd = args.net
num2test = args.num

ImageFile.LOAD_TRUNCATED_IMAGES=True

with open('./savedoc/labels.pkl','rb') as f:
    labellst = pickle.load(f)


def ttsfm(label):
    return labellst.index(label)


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
val_batch = 128 


valcls = testClothes(valroot, tsfm, ttsfm)
valdata = DataLoader(dataset=valcls, batch_size=val_batch, shuffle=True)


# test
net = resnet18()
net.fc = nn.Linear(512, numcls, True)
net.load_state_dict(torch.load(netmd))
if is_cuda: net = net.cuda()
net.eval()

gtlst = []
predlst = []
pathlst = []
for idx, data in enumerate(valdata):
    imgs, labels, impaths = data
    if is_cuda: imgs, labels = imgs.cuda(), labels.cuda()
    outputs = net(imgs)
    maxvalue = torch.max(outputs, dim=1)[-1]
    pred =  maxvalue.cpu().numpy() 
    gt = labels.cpu().numpy()
    gtlst += list(gt)
    predlst += list(pred)
    pathlst += list(impaths)
    gtarr = np.array(gtlst)
    predarr = np.array(predlst)
    numT = (gtarr==predarr).sum()
    print('iteration:', idx+1, f'the precision of {idx+1} is:', f'{100*numT/len(gtlst):.3f}%')
    if (idx + 1) == num2test:
        break
saved_file = {}
saved_file['gt'] = gtlst
saved_file['pred'] = predlst
saved_file['paths'] = pathlst
with open('./savedoc/testout.pkl', 'wb') as f:
    pickle.dump(saved_file, f)
