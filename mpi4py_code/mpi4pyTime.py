#passRandomDraw.py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_linalg
import time
import os
import sys
from sklearn.preprocessing import normalize

# ff = 'abalone_scale.txt'
# ff = 'w4a.txt'
threads = int(sys.argv[1])
ff = sys.argv[2]
print("The number of threads we are making the file for is " + str(threads) + ".")

# Just a useful function.
def partition(lst, n):
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in range(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in range(n)]

def partition_ind(lst, n):
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in range(n+1)]
    return [list(range(indices[i],indices[i+1])) for i in range(n)]

def summon_files():

    # Data generation and distribution 
    m = np.zeros([1],dtype=int)
    n = np.zeros([1],dtype=int)
    m[0] = np.load('./m_data/'+'m_'+ff+'.npz')['m']
    n[0] = np.load('./n_data/'+'n_'+ff+'.npz')['n']
    n = n[0]
    m = m[0]

    print("m = " + str(m) + ", n = " + str(n))

    b = np.load('./b_data/'+'b_'+ff+'.npz')['b']
    A = sp.load_npz('./A_data/'+ff+'.npz').tolil()

    print(A.shape)
    
    ide = sp.lil_matrix((n,m),dtype=float)
    print(ide.shape)
   
    for i in range(m):
        ide[i,i] = 5.0e-1
        
    A = A + ide

    # A = A - np.tile(np.mean(A,axis=0),(n[0],1))
    A = normalize(A, axis=0, norm='l2').tolil()
    
    # b = b - np.mean(b) 
    A_data = A.data.tolist()
    A_data_partitioned = partition(A_data, threads)

    A_T_data = A.T.data.tolist()
    A_T_data_partitioned = partition(A_T_data, threads)
   
    A_rows = A.rows.tolist()   
    A_rows_partitioned = partition(A_rows, threads)

    A_T_rows = A.T.rows.tolist()
    A_T_rows_partitioned = partition(A_T_rows, threads)
    A_T_rows_ind = partition_ind(A_T_rows, threads)

    b = b.tolist()
    b_partitioned = partition(b, threads)
    b_T_partitioned = partition(b, 1)

    for i in range(threads):
        f = open("ad" + str(i) + ".txt", "w")
        length_A_data = len(A_data_partitioned[i])
        f.write(str(length_A_data) + "\n")
        for j in range(length_A_data):
            for k in range(len(A_data_partitioned[i][j])):
                f.write(str(A_data_partitioned[i][j][k]) + "\n")
            f.write("---\n")

        f = open("ar" + str(i) + ".txt", "w")
        length_A_rows = len(A_rows_partitioned[i])
        f.write(str(length_A_rows) + "\n")
        for j in range(length_A_rows):
            for k in range(len(A_rows_partitioned[i][j])):
                f.write(str(A_rows_partitioned[i][j][k]) + "\n")
            f.write("---\n")

        f = open("b" + str(i) + ".txt", "w")
        length_b = len(b_partitioned[i])
        f.write(str(length_b) + "\n")
        for j in range(length_b):
            f.write(str(b_partitioned[i][j]) + "\n")

        f = open("adT" + str(i) + ".txt", "w")
        length_A_T_data = len(A_T_data_partitioned[i])
        f.write(str(length_A_T_data) + "\n")
        for j in range(length_A_T_data):
            for k in range(len(A_T_data_partitioned[i][j])):
                f.write(str(A_T_data_partitioned[i][j][k]) + "\n")
            f.write("---\n")

        f = open("arT" + str(i) + ".txt", "w")
        length_A_T_rows = len(A_T_rows_partitioned[i])
        f.write(str(length_A_T_rows) + "\n")
        for j in range(length_A_T_rows):
            for k in range(len(A_T_rows_partitioned[i][j])):
                f.write(str(A_T_rows_partitioned[i][j][k]) + "\n")
            f.write("---\n")

        f = open("arI" + str(i) + ".txt", "w")
        length_A_T_rows = len(A_T_rows_ind[i])
        f.write(str(length_A_T_rows) + "\n")
        for j in range(length_A_T_rows):
            f.write(str(A_T_rows_ind[i][j]) + "\n")

        if i == 0:
            f = open("bT" + str(i) + ".txt", "w")
            length_b_T = len(b_T_partitioned[i])
            f.write(str(length_b_T) + "\n")
            for j in range(length_b_T):
                f.write(str(b_T_partitioned[i][j]) + "\n")

    f = open("demomn.txt", "w")
    f.write(str(m) + "\n")
    f.write(str(n) + "\n")

#4177x8
#6x4

def generate_matrix():
    # to_ret = np.array([[3, 0, 0, 0], [0, 6, 7, 0], [8, 0, 0, 5], [2, 0, 1, 0], [0, 4, 0, 9], [1, 2, 3, 4]])
    to_ret = np.random.rand(6, 4)
    while np.linalg.matrix_rank(to_ret) != 4:
        to_ret = np.random.rand(6, 4)
    sparse_to_ret = sp.csc_matrix(to_ret)
    sp.save_npz('./A_data/'+ff+'.npz', sparse_to_ret)
    # b = np.array([1, 2, 3, 4, 5, 6])
    b = np.random.rand(6, )
    np.savez('./b_data/'+'b_'+ff+'.npz', b=b)

summon_files()
 