# coding:utf8
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from PIL import ImageFile
import pickle
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES=True

def myloader(impath):
    with Image.open(impath) as img:
        return img.convert('RGB')

def root_parse(root):
    with open(root, 'rb') as f:
        mlabels = pickle.load(f)
    mlabels = { i:mlabels[i] for i in mlabels if len(mlabels[i])>1}
    with open('./savedoc/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    imroot = Path('/home/feijiang/datasets/images')
    imgs = []
    for img, label in mlabels.items():
        labelname = labels[label[0]]
        imgpath = imroot/labelname/img
        ohlabel = np.zeros(len(labels))
        ohlabel[np.array(label)] = 1 
        imgs.append([imgpath, ohlabel])
    return imgs


class MLClothes(Dataset):
    def __init__(self, root, transform=None, target_transform=None, img_loader=myloader):
        self.tsfm = transform
        self.ttsfm = target_transform
        self.loader = img_loader
        self.imgs = root_parse(root)

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(impath)
        if self.tsfm is not None:
            img = self.tsfm(img)
        if self.ttsfm is not None:
            label = self.ttsfm(label)
        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root = './savedoc/mlabels_img.pkl'
    sets = MLClothes(root)
    for i, j in sets:
        print(i, type(j))
