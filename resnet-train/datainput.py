# coding:utf8
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

def myloader(impath):
    with Image.open(impath) as img:
        return img.convert('RGB')

def root_sparse(root):
    root = Path(root)
    all_imgs = []
    for p in root.iterdir():
        for p1 in p.iterdir():
            label = str(p1).split('/')[-2]
            all_imgs.append((str(p1), label))
    return all_imgs



class Clothes(Dataset):
    def __init__(self, root, transform=None, target_transform=None, img_loader=myloader):
        self.imgs = root_sparse(root)
        self.root = root
        self.tsfm = transform
        self.ttsfm = target_transform
        self.loader = img_loader

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
