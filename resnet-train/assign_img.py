# coding:utf8
import pickle
import shutil
from pathlib import Path
import argparse as agp
import numpy as np

np.random.seed(0)

parser = agp.ArgumentParser(description='Divide tested images into different folders')
parser.add_argument('--lnum', type=int,  default=-1, help='the label num to assign, if negative, all the labels')
parser.add_argument('--num', type=int, default=100, help='the image number to be put')
args = parser.parse_args()


labelnum = args.lnum
num = args.num

root = Path('./testedimg')
if not root.is_dir():
    root.mkdir()

with open('./savedoc/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
with open('./savedoc/testout.pkl', 'rb') as f:
    testout = pickle.load(f)

if labelnum >= 0:
    labelname = labels[labelnum]
    lroot = root/labelname
    if not lroot.is_dir(): lroot.mkdir()
    arrgt = np.array(testout['gt'])
    arrpred = np.array(testout['pred'])
    arrpaths = np.array(testout['paths'])
    imgs = arrpaths[arrpred==labelnum]
    np.random.shuffle(imgs)
    bflag = 0
    for orgpath in imgs:
        shutil.copy(orgpath, lroot)
        bflag += 1
        if bflag == num:
            break 
else:
    for idx in range(len(labels)):
        labelname = labels[idx]
        print(f'{idx+1}, processing {labelname}')
        lroot = root/labelname
        if not lroot.is_dir(): lroot.mkdir()
        arrgt = np.array(testout['gt'])
        arrpred = np.array(testout['pred'])
        arrpaths = np.array(testout['paths'])
        imgs = arrpaths[arrpred==idx]
        np.random.shuffle(imgs)
        bflag = 0
        for orgpath in imgs:
            shutil.copy(orgpath, lroot)
            bflag += 1
            if bflag == num:
                break 
