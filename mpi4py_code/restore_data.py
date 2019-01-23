import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_linalg
import sys
sys.path.insert(0, './libsvm-3.22/python/')
#from svmutil import *
from os import listdir
from os.path import isfile, join

def power_two(n):
    return n.bit_length() - 1

#sys.path = [os.path.dirname(os.path.abspath(__file__))] + sys.path

def svm_read_problem(data_file_name):
    """
    svm_read_problem(data_file_name) -> [y, x]

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []
    for line in open(data_file_name):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        xi = {}
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind)] = float(val)
        prob_y += [float(label)]
        prob_x += [xi]
    return (prob_y, prob_x)

mypath = './datasets/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for ff in onlyfiles:
    print(ff, flush=True)

    path_to_data = mypath + ff
    b, data = svm_read_problem(path_to_data)
    b = np.asarray(b)

    m = -1
    n = len(data)
    for i in range(len(data)):
        if len(data[i]) == 0:
            continue
        for key in data[i].keys():
            if key > m:
                m = key

    A = sp.lil_matrix((n,m),dtype=float)
    for i in range(len(data)):
        for key in data[i].keys():
            A[i,key-1] = data[i][key]

    A = A.tocsr()
    np.savez('./b_data/'+'b_'+ff,b = b)
    np.savez('./n_data/'+'n_'+ff,n = n)
    np.savez('./m_data/'+'m_'+ff,m = m)
    sp.save_npz('./A_data/'+ff,A)
    
    f = open('./m_powersOf2/'+ff,'w')
    f.write(str(np.power(2,power_two(m))))
    f.close()
    f = open('./n_powersOf2/'+ff,'w')
    f.write(str(np.power(2,power_two(n))))
    f.close()

