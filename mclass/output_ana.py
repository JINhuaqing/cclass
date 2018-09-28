# coding:utf8
import pickle
import numpy as np


np.set_printoptions(precision=4)
with open('./savedoc/myfile.pkl', 'rb') as f:
    data = pickle.load(f)

def diff(x):
    x1 = np.concatenate([x[1:], [0]])
    return (x-x1)[:-1]

data0 = [i for i, j in data]
data1 = [j for i, j in data]
data0 = np.concatenate(data0)
data1 = np.concatenate(data1)

stat = []
for pre, gt in zip(data0, data1):
    gtlabels = np.where(gt==1)[0]
    argidx = np.argsort(-pre)
    sortpre = pre[argidx]
    dif = diff(sortpre)[:5]
    idxmaxgap = np.argmax(dif[1:])+2
    pred = argidx[:idxmaxgap]
    print(idxmaxgap, len(gtlabels))
    stat.append(idxmaxgap==len(gtlabels))
print(np.sum(stat)/len(stat))
