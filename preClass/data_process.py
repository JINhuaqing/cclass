# coding:utf8
import numpy as np
from pathlib import Path
import pickle
from PIL import Image


root = Path('/home/feijiang/datasets/images')

rgb = [] 
outflag = 0
for p1 in root.iterdir():
    lst = list(p1.iterdir())[:20]
    np.random.shuffle(lst)
    for p2 in lst: 
        try:
            img = np.array(Image.open(p2))
            add3 = np.mean(img, axis=(0, 1))
        except IOError:
            add3 = np.array([0, 0, 0])
        except IndexError:
            add3 = np.array([0, 0, 0])
        if add3.sum() != 0:
            rgb.append(add3) 
            outflag += 1
        print(outflag)
rgb = np.array(rgb)
channelmean = np.mean(rgb, axis=0)
with open('./savedoc/mean.txt', 'w') as f:
    f.write(str(channelmean))
