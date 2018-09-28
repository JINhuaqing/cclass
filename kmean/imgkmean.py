# coding:utf8
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import pickle
from numpy.random import shuffle


class KmeansIm():
    def __init__(self, impath, nclusters=3):
        self.nclusters = nclusters
        self.img = cv2.imread(impath, 0)
        self.d1img = self.img.reshape(-1, 1) 
        self.labels = None  
        self.v = None
  
    def kmeans(self):
        result = KMeans(n_clusters=3, random_state=0).fit(self.d1img)
        if self.labels is None: self.labels = result.labels_
        
    def imvar(self):
        c1 = self.d1img[self.labels==0].reshape(-1)
        c2 = self.d1img[self.labels==1].reshape(-1)
        c3 = self.d1img[self.labels==2].reshape(-1)
        self.v = np.var(c1) + np.var(c2) + np.var(c3)
        
    def init(self):
        self.kmeans()
        self.imvar()


if __name__ == '__main__':
    root = Path('/home/feijiang/datasets/imgval')
    varlst = []
    num = 40 
    clslst = list(root.iterdir())
    shuffle(clslst)
    for idx, p in enumerate(clslst):
        cflag = 0
        pathlst = list(p.iterdir())
        shuffle(pathlst)
        for idx0, pp in enumerate(pathlst):
            try:
                im = KmeansIm(str(pp))
            except AttributeError:
                continue
            im.init()
            varlst.append([str(pp), im.v])
            cflag += 1
            print(f'class {idx+1}, img {idx0+1},  {str(pp)} finished!')
            if cflag == num:
                break
    with open('./savedoc/imvars.pkl', 'wb') as f:
        pickle.dump(varlst, f)

