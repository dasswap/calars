#passRandomDraw.py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_linalg
from mpi4py import MPI
import time
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, './libsvm-3.22/python/')
from svmutil import *
from bLARS_tournament import bLARS_tournament

# Fix seed 
np.random.seed(seed=10)
#np.random.RandomState = 10

# Just a useful function.
def diff(a, b):
        b = set(b)
        return [aa for aa in a if aa not in b]
    
def partition(lst, n):
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in range(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in range(n)],[list(range(indices[i],indices[i+1])) for i in range(n)]
    
def solve_linear(L,b):
    y = sp_linalg.solve_triangular(L,b,lower=True)
    return sp_linalg.solve_triangular(L,y,trans='T',lower=True)

def solve_quadratic(rhs):
    n = rhs.shape[0]
    Omega = np.zeros((n,n))
    
    if n > 1:
        for i in reversed(range(n)):
            for j in reversed(range(i+1)):
                if i == j:
                    if i < n-1:
                        Omega[i,j] = np.sqrt(rhs[i,j] - np.sum(np.square(Omega[i+1:n,j])))
                    else:
                        Omega[i,j] = np.sqrt(rhs[i,j])
                else:
                    if i < n-1:
                        Omega[i,j] = (rhs[i,j] - np.prod(Omega[i+1,0:i+1]))/Omega[i,i]
                    else:
                        Omega[i,j] = rhs[i,j]/Omega[i,i]
    else:
        Omega[0,0] = np.sqrt(rhs[0,0])
        
    return Omega

# Initialize MPI properties
comm = MPI.COMM_WORLD
size = comm.Get_size()
p = comm.Get_rank()

if p == 0:
    print("Size: ", size)

# Data generation and distribution
m = np.zeros([1],dtype=int)
n = np.zeros([1],dtype=int)

if p == 0:
    #path_to_data = input("Write the path to the data without quotes \n")
    #path_to_data = './libsvm-3.22/heart_scale'
    path_to_data = "./libsvm-3.22/heart_scale"
    b, data = svm_read_problem(path_to_data)
    b = np.asarray(b)
        
    m[0] = -1
    n[0] = len(data)
    for i in range(len(data)):
        for key in data[i].keys():
            if key > m:
                m[0] = key
    
    A = sp.lil_matrix((n[0],m[0]),dtype=float)
    for i in range(len(data)):
        for key in data[i].keys():
            A[i,key-1] = data[i][key]
    
#    A = A - np.tile(np.mean(A,axis=0),(n[0],1))
    A = normalize(A, axis=0, norm='l2').tolil()
#    b = b - np.mean(b)    
    A_T_data = A.T.data.tolist()
    A_T_data_partitioned, A_T_data_ind= partition(A_T_data, size)
    A_T_rows = A.T.rows.tolist()
    A_T_rows_partitioned, A_T_rows_ind = partition(A_T_rows, size)
    
    b = np.array(b)
    
comm.Bcast(m, root=0)
comm.Bcast(n, root=0)

n = n[0]
m = m[0]

block = 1
iterations = int(np.floor(m/block))

if p!=0:
    A_T_data_partitioned = None
    A_T_rows_partitioned = None
    A_T_data_ind = None
    A_T_rows_ind = None
    b = np.zeros(n)

A_T_data_local = comm.scatter(A_T_data_partitioned, root=0)
A_T_rows_local = comm.scatter(A_T_rows_partitioned, root=0)
A_T_data_ind_local = comm.scatter(A_T_data_ind, root=0)
A_T_rows_ind_local = comm.scatter(A_T_rows_ind, root=0)
comm.Bcast(b, root=0)

m_local = len(A_T_data_local)

A_local = sp.lil_matrix((m_local,n),dtype=float)
for row in range(m_local):
    ct = 0
    for col in A_T_rows_local[row]:
        A_local[row,col] = A_T_data_local[row][ct]
        ct = ct + 1
A_local = A_local.T.tocsr()

length_of_tree = int(np.log2(size))+1

tree_list = [[] for i in range(length_of_tree)]

for i in range(length_of_tree):
    tree_list[i] = list(range(0,size,np.power(2,i)))
    
if length_of_tree == 0:
    length_of_tree = 1
    tree_list = [[0]]

# Start measuring time
if p == 0:
    start = time.time()

b_sol = np.zeros(n)
A_hallow = []
A_hallow_p0 = []

#iterations = 51

for iteration in range(iterations):
    #print("\nProcessor: ",p," iteration: ", iteration)
    
    for i in range(length_of_tree):
        #print("Processor: ",p," i: ", i)
        
        for j in tree_list[i]:
            #print("Processor: ",p," j: ", j)
            
            if p == j:
                
                A_node = A_local.copy()
                A_T_rows_ind_node_local = A_T_rows_ind_local[:]
                
                if i > 0:
                    B2_actual = np.zeros(block,dtype=int)
                    comm.Recv(B2_actual, source=p+np.power(2,i-1), tag=1)
                    
                    A_B2_lengths = np.zeros(3,dtype=int)
                    comm.Recv(A_B2_lengths, source=p+np.power(2,i-1), tag=2)
                    
                    A_data_B2 = np.zeros(A_B2_lengths[0])
                    A_indices_B2 = np.zeros(A_B2_lengths[1],dtype=np.int32)
                    A_indptr_B2 = np.zeros(A_B2_lengths[2],dtype=np.int32)
                    
                    comm.Recv(A_data_B2, source=p+np.power(2,i-1), tag=3)
                    comm.Recv(A_indices_B2, source=p+np.power(2,i-1), tag=4)
                    comm.Recv(A_indptr_B2, source=p+np.power(2,i-1), tag=5)
                    
                    A_B2 = sp.csc_matrix((A_data_B2, A_indices_B2, A_indptr_B2), shape=(block, n)).T
                    
                    A_node = sp.hstack([A_node,A_B2]).tocsr()
                    
                    A_T_rows_ind_node_local.extend(B2_actual)
                    
                    #B1_actual = np.zeros(block,dtype=int)
                    #comm.Irecv(B1_actual, source=p, tag=6)
                    
                    if B1_actual.shape[0] > 0:
                    
                        # XXX the code below might create problems for blocks larger than 1.
                        check_it = np.where(A_T_rows_ind_local == B1_actual[0])[0].shape[0]
                        if check_it == 0:

                            #A_B1_lengths = np.zeros(3,dtype=int)
                            #comm.Irecv(A_B1_lengths, source=p, tag=7)

                            #A_data_B1 = np.zeros(A_B1_lengths[0])
                            #A_indices_B1 = np.zeros(A_B1_lengths[1],dtype=np.int32)
                            #A_indptr_B1 = np.zeros(A_B1_lengths[2],dtype=np.int32)

                            #comm.Irecv(A_data_B1, source=p, tag=8)
                            #comm.Irecv(A_indices_B1, source=p, tag=9)
                            #comm.Irecv(A_indptr_B1, source=p, tag=10)

                            A_B1 = sp.csc_matrix((A_data_B1, A_indices_B1, A_indptr_B1), shape=(block, n)).T

                            A_node = sp.hstack([A_node,A_B1]).tocsr()

                            A_T_rows_ind_node_local.extend(B1_actual)
                    
                m_node = A_node.shape[1]
                
                if np.mod(j,np.power(2,i+1)) == 0:
                    
                    #print("Processor: ",p," A_hallow: ", A_hallow)
                    #print("Processor: ",p," A_hallow_p0: ", A_hallow_p0)
                    b_sol_,A_hallow_,B1 = bLARS_tournament(A_node,b,m_node,n,1,A_hallow_p0[:],np.copy(b_sol),p)
                    #print("Processor: ",p," A_T_rows_ind_node_local: ", A_T_rows_ind_node_local)
                    #print("Processor: ",p," B1: ", B1)
                    B1_actual = np.asarray(A_T_rows_ind_node_local)[B1]
                    #print("Processor: ",p," B1_actual: ", B1_actual)
                    
                    #if np.mod(j,np.power(2,i+1)) != 0:
                    #    B2 = B1
                    #    B2_actual = B1_actual
                    #    print("Processor: ",p," B2: ", B2)
                    #    print("Processor: ",p," B2_actual: ", B2_actual)
                    #    comm.Send(B2_actual, dest=p-np.power(2,i), tag=1)
                    #
                    #    A_B2 = A_node.tocsc()[:,B2].tocsr()
                    #    A_data_B2 = A_B2.data
                    #    A_indices_B2 = A_B2.indices
                    #    A_indptr_B2 = A_B2.indptr
                    #
                    #    A_B2_lengths = np.asarray([A_data_B2.shape[0],A_indices_B2.shape[0],A_indptr_B2.shape[0]])
                    #
                    #    comm.Send(A_B2_lengths, dest=p-np.power(2,i), tag=2) # consider combining these
                    #    comm.Send(A_data_B2, dest=p-np.power(2,i), tag=3)
                    #    comm.Send(A_indices_B2, dest=p-np.power(2,i), tag=4)
                    #    comm.Send(A_indptr_B2, dest=p-np.power(2,i), tag=5)
                    
                    #comm.Isend(B1_actual, dest=p, tag=6)
                    
                    if i != length_of_tree - 1:
                        
                        #print("Processor: ",p," B1_actual.shape[0]: ", B1_actual.shape[0])
                        
                        if B1_actual.shape[0] > 0:
                    
                            # XXX the code below might create problems for blocks larger than 1.
                            check_it = np.where(A_T_rows_ind_local == B1_actual[0])[0].shape[0]
                            if check_it == 0:

                                A_B1 = A_node.tocsc()[:,B1].tocsr()
                                A_data_B1 = A_B1.data
                                A_indices_B1 = A_B1.indices
                                A_indptr_B1 = A_B1.indptr

                                A_B1_lengths = np.asarray([A_data_B1.shape[0],A_indices_B1.shape[0],A_indptr_B1.shape[0]])

                                #comm.Isend(A_B1_lengths, dest=p, tag=7) # consider combining these
                                #comm.Isend(A_data_B1, dest=p, tag=8)
                                #comm.Isend(A_indices_B1, dest=p, tag=9)
                                #comm.Isend(A_indptr_B1, dest=p, tag=10)
                        
                else:
                    #print("Processor: ",p," A_hallow: ", A_hallow)
                    #print("Processor: ",p," A_hallow_p0: ", A_hallow_p0)
                    #print("Processor: ",p," A_T_rows_ind_local: ", A_T_rows_ind_local)
                    b_sol_,A_hallow_,B2 = bLARS_tournament(A_node,b,m_node,n,block,A_hallow_p0[:],np.copy(b_sol),p)
                    #print("Processor: ",p," B2: ", B2)
                    B2_actual = np.asarray(A_T_rows_ind_node_local)[B2]
                    #print("Processor: ",p," B2_actual: ", B2_actual)
                    comm.Send(B2_actual, dest=p-np.power(2,i), tag=1)

                    A_B2 = A_node.tocsc()[:,B2].tocsr()
                    A_data_B2 = A_B2.data
                    A_indices_B2 = A_B2.indices
                    A_indptr_B2 = A_B2.indptr
                    
                    A_B2_lengths = np.asarray([A_data_B2.shape[0],A_indices_B2.shape[0],A_indptr_B2.shape[0]])
                    
                    comm.Send(A_B2_lengths, dest=p-np.power(2,i), tag=2) # consider combining these
                    comm.Send(A_data_B2, dest=p-np.power(2,i), tag=3)
                    comm.Send(A_indices_B2, dest=p-np.power(2,i), tag=4)
                    comm.Send(A_indptr_B2, dest=p-np.power(2,i), tag=5)
    if p == 0:
        #print("Processor: ",p," B1: ", B1)
        ct = 0
        ct2 = 0
        for col in B1:
            if col > m_local-1:
                A_local = sp.hstack([A_local,A_node.tocsc()[:,col]]).tocsr()
                if block == 1:
                    A_T_rows_ind_local.extend(B1_actual)
                else:
                    A_T_rows_ind_local.extend(B1_actual[ct])
                A_hallow_p0.extend([m_local + ct2])
                ct2 = ct2 + 1
            else:
                A_hallow_p0.extend([col])
            ct = ct + 1
        
        m_local = m_local + ct2
        
        #print("Processor: ",p," A_T_rows_ind_local: ", A_T_rows_ind_local)         
        #print("Processor: ",p," A_local.shape: ", A_local.shape) 
        #print("Processor: ",p," A_node.shape: ", A_node.shape)
              
        A_hallow.extend(B1_actual)
        
        #print("Processor: ",p," A_hallow_p0: ", A_hallow_p0) 
        #print("Processor: ",p," A_hallow: ", A_hallow) 
        
        b_sol = np.asarray(b_sol_)

        A_B1 = A_node.tocsc()[:,B1].tocsr()
        A_data_B1 = A_B1.data
        A_indices_B1 = A_B1.indices
        A_indptr_B1 = A_B1.indptr
                    
        A_B1_lengths = np.array([A_data_B1.shape[0],A_indices_B1.shape[0],A_indptr_B1.shape[0]],dtype=int)
    else:
        B1_actual = np.zeros(block,dtype=int)
        A_B1_lengths = np.zeros(3,dtype=int)
    
    comm.Bcast(B1_actual, root=0) # consider combining these
    comm.Bcast(A_B1_lengths, root=0)
    
    if p != 0:
        A_data_B1 = np.zeros(A_B1_lengths[0])
        A_indices_B1 = np.zeros(A_B1_lengths[1],dtype=np.int32)
        A_indptr_B1 = np.zeros(A_B1_lengths[2],dtype=np.int32)
    
    comm.Bcast(A_data_B1, root=0)
    comm.Bcast(A_indices_B1, root=0)
    comm.Bcast(A_indptr_B1, root=0)
    
    comm.Bcast(b_sol, root=0)
    
    if p != 0:
        #print("Processor: ",p," ENTERED")
        A_hallow.extend(B1_actual)
        A_B1 = sp.csc_matrix((A_data_B1, A_indices_B1, A_indptr_B1), shape=(block, n)).T

        ct = 0
        for col in B1_actual:
            check_it = np.where(A_T_rows_ind_local == col)[0].shape[0]
            if check_it == 0:
                A_local = sp.hstack([A_local,A_B1[:,ct]]).tocsr()
                A_hallow_p0.extend([m_local+ct])
                ct = ct + 1
                m_local = m_local + ct
                A_T_rows_ind_local.extend([col])
            else:
                idx = np.where(A_T_rows_ind_local == col)[0][0]
                #print("Processor: ",p," idx: ", idx)
                A_hallow_p0.extend([idx])

if p == 0:
    print("A_hallow: ",A_hallow)
    #print("A_hallow_p0: ",A_hallow_p0)
    end = time.time()
    print("Time: ", end-start)