# coding:utf8
import os
from torchvision import transforms as tsfms
from PIL import Image 
from PIL import ImageFile
from pathlib import Path
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES=True

parser = argparse.ArgumentParser(description='remove the problem images')
parser.add_argument('--root', type=str, default='/home/feijiang/datasets/images',  help='the image root')

args = parser.parse_args()
root = Path(args.root)

tsfm = tsfms.Compose([tsfms.Resize([224, 224]), tsfms.ToTensor(), tsfms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.1, 0.1, 0.1])])

cflag = 0
aflag = 0
for pp in root.iterdir():
    for img in pp.iterdir():
        aflag += 1
        try:
            with Image.open(img) as f:
                im = f.convert('RGB')
            im = tsfm(im)
        except KeyboardInterrupt:
            os._exit(0)
        except:
            os.remove(img)
            cflag += 1
            print(cflag, img)
        print('all images:', aflag, cflag)
