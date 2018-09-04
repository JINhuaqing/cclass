# coding:utf8
import numpy as np
from pathlib import Path
import pickle
from PIL import Image


root = Path('/home/feijiang/datasets/images')

rgbmean = [] 
rgbstd = []
outflag = 0
for p1 in root.iterdir():
    lst = list(p1.iterdir())[:20]
    np.random.shuffle(lst)
    for p2 in lst: 
        try:
            with Image.open(p2) as f:
                img = np.array(f)
            mean = np.mean(img, axis=(0, 1))
            std = np.std(img, axis=(0, 1))
        except ValueError:
            mean = np.array([0, 0, 0])
            std = np.array([0, 0, 0])
        except IndexError:
            mean = np.array([0, 0, 0])
            std = np.array([0, 0, 0])
        if mean.sum() != 0:
            rgbmean.append(mean)
            rgbstd.append(std)
            outflag += 1
        print(outflag)
channelmean = np.mean(rgbmean, axis=0)
channelstd = np.std(rgbstd, axis=0)
with open('./savedoc/mean.pkl', 'wb') as fmean:
    pickle.dump(channelmean, fmean)
with open('./savedoc/std.pkl', 'wb') as fstd:
    pickle.dump(channelstd, fstd)
