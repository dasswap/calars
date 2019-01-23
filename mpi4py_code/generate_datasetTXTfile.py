import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_linalg
import sys
sys.path.insert(0, './libsvm-3.22/python/')
#from svmutil import *
from os import listdir
from os.path import isfile, join

mypath = './datasets/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

f = open('datasets.txt','w')

for ff in onlyfiles:
    print(ff)
    f.write(ff)
    f.write('\n')
    
f.close()
    