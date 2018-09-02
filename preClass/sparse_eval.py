# coding:utf8
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
plt.switch_backend('agg')


def clist_th(lst, thre=2):
    ct = Counter(lst)
    return [i[0] for i in ct.items() if i[1]<=thre]

# the file root
root = './savedoc/tmp.txt'
root = Path(root)
assert root.is_file(), "the file does not exist"

# get the file
with open(root) as f:
    f = f.readlines()

# handle the f
f = [[i.split('jpg')[0].split('/')[-2], i.split('jpg')[1].strip()] for i in f]
gts = [i[0] for i in f]
pres = [i[1].split(' ')[0] for i in f]
nppres = np.array(pres)
npgts = np.array(gts)
types = set(gts)

for tp in types:
    tmp = nppres[npgts==tp]
    litems = clist_th(tmp, 100)
    tmp = [i for i in tmp if i not in litems]
    print(tp)
    plt.hist(tmp, bins=len(set(tmp)))
    plt.savefig(f'./savedoc/{tp}.jpg')
    plt.close()
