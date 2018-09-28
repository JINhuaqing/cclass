# coding:utf8
import numpy as np
import pickle
from pathlib import Path
import shutil

root = Path('./kmean/imgs')
if not root.is_dir(): root.mkdir()

with open('./savedoc/imvars.pkl', 'rb') as f:
    data = pickle.load(f)

X = [i[0] for i in data]
Y = [i[-1] for i in data]
Xnp = np.array(X)
Ynp = np.array(Y)
qt = np.percentile(Y, 10)

rest = Xnp[Ynp<qt]

for i in rest:
    shutil.copy(i, root)
