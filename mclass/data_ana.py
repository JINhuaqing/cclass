# coding:utf8
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
from collections import defaultdict as ddict

root = Path('/home/feijiang/datasets/images')
assert root.is_dir(), 'path does not exist'

with open('./savedoc/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# record the label for every images
imgdic = ddict(list) 
for idx, p in enumerate(root.iterdir()):
    print(idx+1, str(p).split('/')[-1])
    for pp in p.iterdir():
        strp = str(pp)
        label, name = strp.split('/')[-2:]
        imgdic[name].append(labels.index(label))

with open('./savedoc/mlabels_img.pkl', 'wb') as f:
    pickle.dump(imgdic, f)
