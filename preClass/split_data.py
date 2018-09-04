# coding:utf8
from pathlib import Path
import numpy as np
import numpy.random as npr
import shutil

roottrain = Path('/home/feijiang/datasets/imgtrain/')
roottest = Path('/home/feijiang/datasets/imgtest/')
rootval = Path('/home/feijiang/datasets/imgval/')
root = Path('/home/feijiang/datasets/images/')

trainratio = 0.8
testratio= 0.1


for i, pp in enumerate(root.iterdir()):
    dirname = str(pp).split('/')[-1]
    traintmp = roottrain/dirname
    valtmp = rootval/dirname
    testtmp = roottest/dirname
    if not traintmp.is_dir(): traintmp.mkdir() 
    if not testtmp.is_dir(): testtmp.mkdir() 
    if not valtmp.is_dir(): valtmp.mkdir() 
    allimages = list(pp.iterdir())
    allimgs = [ str(i).split('/')[-1] for i in allimages]
    npr.shuffle(allimgs)
    
    trainnum, testnum = int(len(allimgs)*trainratio), int(len(allimgs)*testratio)
    trainlst, testlst, vallst = allimgs[:trainnum], allimgs[trainnum:trainnum+testnum], allimgs[trainnum+testnum:]
    
    for img in pp.iterdir():
        imname = str(img).split('/')[-1]
        if imname in trainlst:
            savedpath = traintmp/imname
        elif imname in testlst:
            savedpath = testtmp/imname
        else:
            savedpath = valtmp/imname
        shutil.copy(img, savedpath)
        print('saved img', i+1, dirname, savedpath)
