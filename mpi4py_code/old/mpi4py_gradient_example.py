#passRandomDraw.py
import numpy as np
from mpi4py import MPI
import time

# Initialize MPI properties
comm = MPI.COMM_WORLD
size = comm.Get_size()
p = comm.Get_rank()

if p == 0:
    print("Size: ", size)

# Data generation and distribution
m = 3
n = 8*1000
n_local = int(n/size)

if p == 0:
    A = np.random.rand(n,m)
    b = np.random.rand(n)
else:
    A = np.empty((n,m))
    b = np.empty(n)

A_local = np.empty((n_local,m))
b_local = np.empty(n_local)

comm.Scatter(A, A_local, root=0) # Consinder combining these two in one.
comm.Scatter(b, b_local, root=0)

# Generate initial solution and distribute it
if p == 0:
    x = np.random.rand(m)
else:
    x = np.empty(m)

comm.Bcast(x, root=0)

if p == 0:
    start = time.time()

# The algorithm starts here
for i in range(1000):
    
    AxMb_local = np.dot(A_local,x) - b_local

    grad_local = np.dot(A_local.T,AxMb_local)

    if p == 0:
        grad = np.zeros(m)
    else:
        grad = np.empty(m)

    comm.Reduce(grad_local, grad, op=MPI.SUM, root=0)

    if p == 0:
        x = x - grad/10000
        print("||grad||_2",np.linalg.norm(grad))

    comm.Bcast(x, root=0)

if p == 0:
    end = time.time()
    print("Time: ", end-start)
