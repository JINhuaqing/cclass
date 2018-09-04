# coding:utf8
from pathlib import Path
import pickle

root = Path('/home/feijiang/datasets/images')

labels = [ str(i).split('/')[-1] for i in root.iterdir()]
output = open('./savedoc/labels.pkl', 'wb')
pickle.dump(labels, output)
output.close()
