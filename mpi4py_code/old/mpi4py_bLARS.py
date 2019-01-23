#passRandomDraw.py
import numpy as np
import scipy.linalg as sp_linalg
from mpi4py import MPI
import time
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, './libsvm-3.22/python/')
from svmutil import *

# Fix seed 
np.random.seed(seed=10)
#np.random.RandomState = 10

# Just a useful function.
def diff(a, b):
        b = set(b)
        return [aa for aa in a if aa not in b]
    
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
m = 6
n = 8*1000
n_local = int(n/size)
block = 3
iterations = int(np.floor(m/block))

if p == 0:
    A = np.random.rand(n,m)
    b = np.random.rand(n)
#    b, A = svm_read_problem('./libsvm-3.22/heart_scale')
    A = A - np.tile(np.mean(A,axis=0),(n,1))
    A = normalize(A, axis=0, norm='l2')
    b = b - np.mean(b)
else:
    A = np.empty((n,m))
    b = np.empty(n)
    
#np.set_printoptions(precision=15)
#print(A)
#print(b)

A_local = np.empty((n_local,m))
b_local = np.empty(n_local)

comm.Scatter(A, A_local, root=0) # Consinder combining these two in one.
comm.Scatter(b, b_local, root=0)

# Start measuring time
if p == 0:
    start = time.time()

b_sol_local = np.empty(n_local)
r_local = b_local

c_local = np.dot(A_local.T,r_local)
if p == 0:
    c = np.zeros(m)
else:
    c = np.empty(m)
comm.Allreduce(c_local, c, op=MPI.SUM)

A_hallow = []
B = np.argsort(-abs(c))[0:block]
c_const = abs(c[B[block-1]])
A_hallow.extend(B)
A_hallow_old = list(A_hallow)
l_A_hallow = len(A_hallow)
l_A_hallow_old = len(A_hallow_old)
A_hallow_c = diff(range(m), A_hallow)

G_local = np.dot(A_local[:,A_hallow].T,A_local[:,A_hallow])
if p == 0:
    G = np.zeros((l_A_hallow,l_A_hallow))
else:
    G = np.empty((l_A_hallow,l_A_hallow))
comm.Reduce(G_local, G, op=MPI.SUM, root=0)

w = np.zeros(l_A_hallow)
h = np.zeros(1)
compact_w_h = np.zeros(l_A_hallow+1)

if p == 0:
    L = np.linalg.cholesky(G)
    s = c[A_hallow] 
    q = solve_linear(L,s)
    h = np.asarray([1/np.sqrt(np.dot(s.T,q))])
    w = h*q
    compact_w_h = np.concatenate((w, h), axis=0)    
    
comm.Bcast(compact_w_h, root=0)

w = compact_w_h[0:l_A_hallow]
h = np.asarray([compact_w_h[l_A_hallow]])

u_local = np.dot(A_local[:,A_hallow],w)

a = np.zeros(m)
a_local = np.dot(A_local.T,u_local)
comm.Allreduce(a_local, a, op=MPI.SUM)
    
gamma = np.zeros(1)   
    
compact_B_gamma = np.zeros(block+1)
    
if p == 0:
    print("c_const: ",c_const)

for i in range(iterations-1):

    if p == 0:
        
        l_A_hallow_c = len(A_hallow_c)
        
        min_pos = np.zeros(l_A_hallow_c)
        for j in range(l_A_hallow_c):
            idx = A_hallow_c[j]
            val1 = (c_const - c[idx])/(h*c_const - a[idx])
            val2 = (c_const + c[idx])/(h*c_const + a[idx])
            min_pos[j] = min(val for val in [val1,val2] if val > 0)
            
        if l_A_hallow_c >= block:
            idx = np.argsort(min_pos)[0:block].tolist()
            B = np.asarray(A_hallow_c)[idx].tolist()
            gamma = min_pos[idx[block-1]]
        else:
            idx = np.argsort(min_pos)[0:l_A_hallow_c].tolist()
            B = np.asarray(A_hallow_c)[idx].tolist()
            gamma = min_pos[idx[l_A_hallow_c-1]]
                
        compact_B_gamma = np.concatenate((np.asarray(B),np.asarray([gamma])),axis=0)  
    
    comm.Bcast(compact_B_gamma, root=0)
    B = compact_B_gamma[0:block].astype(int)
    gamma = np.asarray([compact_B_gamma[block]])
    
    B = [B]
     
    b_sol_local = b_sol_local + np.multiply(gamma,u_local)
    
    c[A_hallow] = c[A_hallow]*(1-gamma*h)
    c[A_hallow_c] = c[A_hallow_c] - gamma*a[A_hallow_c]
    c_const = c_const*(1-gamma*h)
    
    if block > 0:
        B = B[0]
    else:
        if p != 0:
            B = B[0]
    
    A_hallow.extend(B)
    l_A_hallow = len(A_hallow)
    A_hallow_c = diff(range(m), A_hallow)
    l_B = len(B)
    
    A_hallow_t_old_A_B_local = np.dot(A_local[:,A_hallow_old].T,A_local[:,B])
    if p == 0:
        A_hallow_t_old_A_B = np.zeros((l_A_hallow_old,l_B))
    else:
        A_hallow_t_old_A_B = np.empty((l_A_hallow_old,l_B))
    comm.Reduce(A_hallow_t_old_A_B_local, A_hallow_t_old_A_B, op=MPI.SUM, root=0)
    
    A_B_t_A_B_local = np.dot(A_local[:,B].T,A_local[:,B])
    if p == 0:
        A_B_t_A_B = np.zeros((l_B,l_B))
    else:
        A_B_t_A_B = np.empty((l_B,l_B))
    comm.Reduce(A_B_t_A_B_local, A_B_t_A_B, op=MPI.SUM, root=0)    
    
    if p == 0:
        H = np.zeros((l_A_hallow_old,l_B))
        
        for j in range(l_B):
            H[:,j] = sp_linalg.solve_triangular(L,A_hallow_t_old_A_B[:,j],lower=True)
           
        rhs_Omega = A_B_t_A_B - np.dot(H.T,H)
            
        Omega = solve_quadratic(rhs_Omega)
            
        temp4 = np.concatenate((H.T, Omega), axis=1)

        L = np.concatenate((L, np.zeros((l_A_hallow_old,l_B))), axis=1)
        L = np.concatenate((L, temp4), axis=0)
        
        s = c[A_hallow] 
        q = solve_linear(L,s)
        h = np.asarray([1/np.sqrt(np.dot(s.T,q))])
        w = h*q
        compact_w_h = np.concatenate((w, h), axis=0)
    else:
        compact_w_h = np.zeros(l_A_hallow+1)
        
    comm.Bcast(compact_w_h, root=0)

    w = compact_w_h[0:l_A_hallow]
    h = np.asarray([compact_w_h[l_A_hallow]])
    
    u_local = np.dot(A_local[:,A_hallow],w)
    
    a_local = np.dot(A_local.T,u_local)

    comm.Allreduce(a_local, a, op=MPI.SUM)

    A_hallow_old = list(A_hallow)
    l_A_hallow = len(A_hallow)
    l_A_hallow_old = len(A_hallow_old)
    
    if p == 0:
        print("c_const: ",c_const)

if p == 0:
    print("A_hallow: ",A_hallow)
    end = time.time()
    print("Time: ", end-start)