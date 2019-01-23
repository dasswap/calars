#passRandomDraw.py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_linalg
from mpi4py import MPI
import time
from sklearn.preprocessing import normalize

# Just a useful function.
def diff(a, b):
        b = set(b)
        return [aa for aa in a if aa not in b]
    
def partition(lst, n):
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in range(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in range(n)]
    
def solve_linear(L,b):
    y = sp_linalg.solve_triangular(L,b,lower=True)
    return sp_linalg.solve_triangular(L,y,trans='T',lower=True)

def solve_quadratic(rhs):
    n = rhs.shape[0]
    Omega = np.zeros((n,n))
    
    if n > 1:
        for i in range(n):
            for j in range(i,n):
                if i == j:
                    if i != 0:
                        Omega[j,i] = np.sqrt(rhs[j,i] - np.sum(np.square(Omega[j,0:i])))
                    else:
                        Omega[j,i] = np.sqrt(rhs[j,i])
                else:
                    if i != 0:
                        temp = 0
                        for l in range(i):
                            temp = temp + Omega[j,l]*Omega[i,l]
                        Omega[j,i] = (rhs[j,i] - temp)/Omega[i,i]
                    else:
                        Omega[j,i] = rhs[j,i]/Omega[i,i]
    else:
        Omega[0,0] = np.sqrt(rhs[0,0])
        
    return Omega

def parallel_bLARS_svm_data(LARS_Ahallow,block,ff, desired_iters):

    # Initialize MPI properties
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    p = comm.Get_rank()
    og_block = block

    # Data generation and distribution
    m = np.zeros([1],dtype=int)
    n = np.zeros([1],dtype=int)

    if p == 0:
        m[0] = np.load('./m_data/'+'m_'+ff+'.npz')['m']
        n[0] = np.load('./n_data/'+'n_'+ff+'.npz')['n']
        b = np.load('./b_data/'+'b_'+ff+'.npz')['b']
        A = sp.load_npz('./A_data/'+ff+'.npz').tolil()
        
        ide = sp.lil_matrix((n[0],m[0]),dtype=float)
        
        for i in range(m[0]):
            ide[i,i] = 5.0e-1
            
        A = A + ide

        #    A = A - np.tile(np.mean(A,axis=0),(n[0],1))
        A = normalize(A, axis=0, norm='l2').tolil()
        
        #    b = b - np.mean(b) 
        A_data = A.data.tolist()
        A_data_partitioned = partition(A_data, size)
        A_rows = A.rows.tolist()
        A_rows_partitioned = partition(A_rows, size)

        b = b.tolist()
        b_partitioned = partition(b, size)

    comm.Bcast(m, root=0)
    comm.Bcast(n, root=0)

    n = n[0]
    m = m[0]

    iterations = int(np.ceil(min(m,n)/block))
    if min(m, n) % block == 0:
        iterations -= 1
    if (desired_iters < iterations):
        iterations = desired_iters 

    if p!=0:
        A_data_partitioned = None
        A_rows_partitioned = None
        b_partitioned = None

    A_data_local = comm.scatter(A_data_partitioned, root=0)
    A_rows_local = comm.scatter(A_rows_partitioned, root=0)
    b_local = np.asarray(comm.scatter(b_partitioned, root=0))
    n_local = len(A_data_local)

    A_local = sp.lil_matrix((n_local,m),dtype=float)
    for row in range(n_local):
        ct = 0
        for col in A_rows_local[row]:
            A_local[row,col] = A_data_local[row][ct]
            ct = ct + 1
    A_local = A_local.tocsr()

    # Start measuring time
    if p == 0:
        time_total = 0
        start = time.time()

    b_sol_local = np.empty(n_local)
    r_local = b_local

    c_local = A_local.T.dot(r_local)
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
    
    error_Ahallow = []
    linfty_result = [] 
    l2_result = [] 
    time_vector = []

    if p == 0:
        end = time.time()
        time_total  = time_total + (end - start)
        
        linfty_result.extend([c_const])
        l2_result.extend([np.linalg.norm(c,2)])
        
        if len(LARS_Ahallow) != 0:
            if block == 1:
                temp = []
                temp.extend(LARS_Ahallow[0:block])
                if len(diff(temp,A_hallow)) != 0:
                    error_Ahallow.extend([len(diff(temp,A_hallow))])
                else:
                    error_Ahallow.extend([0])
            else:
                if len(diff(LARS_Ahallow[0:block],A_hallow)) != 0:
                    error_Ahallow.extend([len(diff(LARS_Ahallow[0:block],A_hallow))])
                else:
                    error_Ahallow.extend([0])
        start = time.time()

    G_local = A_local[:,A_hallow].T.dot(A_local[:,A_hallow]).toarray()
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
        w = np.multiply(h,q)
        compact_w_h = np.concatenate((w, h), axis=0)    

    comm.Bcast(compact_w_h, root=0)

    w = compact_w_h[0:l_A_hallow]
    h = np.asarray([compact_w_h[l_A_hallow]])

    u_local = A_local[:,A_hallow].dot(w)

    a = np.zeros(m)
    a_local = A_local.T.dot(u_local)
    comm.Allreduce(a_local, a, op=MPI.SUM)

    gamma = np.zeros(1)

    for i in range(iterations):

        l_A_hallow_c = len(A_hallow_c)

        if l_A_hallow_c < block:
            block = l_A_hallow_c

        compact_B_gamma = np.zeros(block+1)

        if p == 0:
            # print("block: ", block)
            # print("l_A_hallow_c: ", l_A_hallow_c)
            min_pos = np.zeros(l_A_hallow_c)
            for j in range(l_A_hallow_c):
                idx = A_hallow_c[j]
                val1 = (c_const - c[idx])/(h*c_const - a[idx])
                val2 = (c_const + c[idx])/(h*c_const + a[idx])
                if val1 < 0:
                    val1 = float("inf")
                if val2 < 0:
                    val2 = float("inf")
                min_pos[j] = min(val1, val2)
            if l_A_hallow_c >= block:
                idx = np.argsort(min_pos)[0:block].tolist()
                B = np.asarray(A_hallow_c)[idx].tolist()
                gamma = 0.9*min_pos[idx[block-1]] # I decrease the original stepsize because of numerical issues.
            else:
                idx = np.argsort(min_pos)[0:l_A_hallow_c].tolist()
                B = np.asarray(A_hallow_c)[idx].tolist()
                gamma = 0.9*min_pos[idx[l_A_hallow_c-1]] # I decrease the original stepsize because of numerical issues.
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
        # print("A_hallow: ", A_hallow)
        l_A_hallow = len(A_hallow)
        A_hallow_c = diff(range(m), A_hallow)
        l_B = len(B)

        ##### The code below is for storing statistics #####
        if p == 0:
            end = time.time()
            time_total  = time_total + (end - start)
            time_vector.extend([time_total])
            
            linfty_result.extend([c_const[0]])
            l2_result.extend([np.linalg.norm(c,2)])
        
            if len(LARS_Ahallow) != 0:
                if len(diff(LARS_Ahallow[0:len(B)*(i+2)],A_hallow)) != 0:
                    error_Ahallow.extend([len(diff(LARS_Ahallow[0:len(B)*(i+2)],A_hallow))])
                else:
                    error_Ahallow.extend([0])
            start = time.time()
        ##### end of computation of statistics #####

        A_hallow_t_old_A_B_local = A_local[:,A_hallow_old].T.dot(A_local[:,B]).toarray()
         # fix all column slices
        shape = A_hallow_t_old_A_B_local.shape
        A_hallow_t_old_A_B_local = A_hallow_t_old_A_B_local.T
        if p == 0:
            A_hallow_t_old_A_B = np.zeros((l_B, l_A_hallow_old))
        else:
            A_hallow_t_old_A_B = np.empty((l_B, l_A_hallow_old))
        comm.Reduce(A_hallow_t_old_A_B_local, A_hallow_t_old_A_B, op=MPI.SUM, root=0)
        if p == 0:
            A_hallow_t_old_A_B = A_hallow_t_old_A_B.T

        A_B_t_A_B_local = A_local[:,B].T.dot(A_local[:,B]).toarray()
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
            w = np.multiply(h,q)
            compact_w_h = np.concatenate((w, h), axis=0)
        else:
            compact_w_h = np.zeros(l_A_hallow+1)

        comm.Bcast(compact_w_h, root=0)

        w = compact_w_h[0:l_A_hallow]
        h = np.asarray([compact_w_h[l_A_hallow]])

        u_local = A_local[:,A_hallow].dot(w)

        a_local = A_local.T.dot(u_local)

        comm.Allreduce(a_local, a, op=MPI.SUM)
        A_hallow_old = list(A_hallow)
        l_A_hallow = len(A_hallow)
        l_A_hallow_old = len(A_hallow_old)

#        print("A_hallow: ",A_hallow)
    if p == 0:
        end = time.time()
        time_total = time_total + (end - start)
        print(str(time_total * 1000) + " milliseconds.")

    return A_hallow, error_Ahallow, linfty_result, l2_result, time_vector
#        end = time.time()
#        print("Time: ", end-start)
