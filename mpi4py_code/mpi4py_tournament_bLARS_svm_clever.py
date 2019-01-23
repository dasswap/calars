#passRandomDraw.py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_linalg
from mpi4py import MPI
import time
from sklearn.preprocessing import normalize
from bLARS_tournament_2 import bLARS_tournament_2

# Fix seed 
np.random.seed(seed=10)
#np.random.RandomState = 10

# Just a useful function.
np.set_printoptions(suppress=True)

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

#, ff
def tourney(LARS_Ahallow, block, ff):

    # Initialize MPI properties
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    p = comm.Get_rank()

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
        A_T_data = A.T.data.tolist()
        A_T_data_partitioned, A_T_data_ind= partition(A_T_data, size)
        A_T_rows = A.T.rows.tolist()
        A_T_rows_partitioned, A_T_rows_ind = partition(A_T_rows, size)

        b = np.array(b)

    comm.Bcast(m, root=0)
    comm.Bcast(n, root=0)

    n = n[0]
    m = m[0]
    
    #print("processor ", p, "n ", n, flush=True)
    #print("processor ", p, "m ", m, flush=True)

    iterations = int(np.ceil(min(m,n)/block))
    if min(m, n) % block != 0:
        iterations += 1
    
    #print("processor ", p, "iterations ", iterations, flush=True)

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
    
    #print("processor ", p, "m_local ", m_local, flush=True)

    A_local = sp.lil_matrix((m_local,n),dtype=float)
    for row in range(m_local):
        ct = 0
        for col in A_T_rows_local[row]:
            A_local[row,col] = A_T_data_local[row][ct]
            ct = ct + 1
    A_local = A_local.T.tocsr()

    # Start measuring time
    if p == 0:
        time_total = 0
        start = time.time()

    time_vector = []
    linfty_result = []
    l2_result = []
    error_Ahallow = []
        
    A_chosen = sp.csr_matrix((n,0))
    b_sol = np.zeros((n,1))
    A_hallow_winners = [] 
    A_hallow = []
    A_hallow_local = []

    L = np.zeros((1,1))

    A_hallow_local_c = np.asarray(range(0,m_local)).tolist()
    
    dim1 = n + 4
    dim2 = block*(A_chosen.shape[1] + block)
    dim = max([dim1,dim2])
    all_srcs = []

    for iteration in range(iterations):
        
        #print("processor ", p, "iteration ", iteration, flush=True)
        #print("processor ", p, "A_local[:,A_hallow_local_c].shape", A_local[:,A_hallow_local_c].shape, flush=True)

        how_many = min(block,len(A_hallow_local_c))
        b_sol_,L_append_1,L_append_2,B,A_B = bLARS_tournament_2(A_chosen,A_local[:,A_hallow_local_c],b,how_many,b_sol[:,0],p,L)

        #print("processor ", p, "A_B.shape", A_B.shape, flush=True)
        
        def my_map(xmem, ymem, dt):

            # print("processor ", p, "Entered", flush=True)
            # print("processor ", p, "dim", dim, flush=True)
            # print("processor ", p, "block", block, flush=True)
            # print("processor ", p, "dst.dtype", dst.dtype, flush=True)
            
            x = np.frombuffer(xmem, dtype=dst.dtype).reshape(dim,block+3)
            y = np.frombuffer(ymem, dtype=dst.dtype).reshape(dim,block+3)

            how_many_1 = x[n+3,0]
            how_many_2 = y[n+3,0]
            how_many_3 = int(min(how_many_1+how_many_2,block))
            #print("processor ", p, "how_many_3", how_many_3, flush=True)
            
            if how_many_3 != 0:
            
                how_many_node_1 = int(min(block,how_many_1))
                how_many_node_2 = int(min(block,how_many_2))

                A_2B = np.concatenate((y[0:n,0:how_many_node_2],x[0:n,0:how_many_node_1]),axis=1)
                proc_v = np.concatenate((y[n,0:how_many_node_2],x[n,0:how_many_node_1]),axis=0)
                indices = np.concatenate((y[n+1,0:how_many_node_2],x[n+1,0:how_many_node_1]),axis=0)
                real_indices = np.concatenate((y[n+2,0:how_many_node_2],x[n+2,0:how_many_node_1]),axis=0)
                #if p == 0:
                #    if iteration > 0:
                #        print("A_chosen.shape", A_chosen.shape, flush=True)
                #        print("A_2B.shape", A_2B.shape, flush=True)
                #        
                #        A_chosen_ = sp.hstack([A_chosen,A_2B]).tocsr()
                #        
                #        G_ = A_chosen_.T.dot(A_chosen_).toarray();
                #
                #        print("G_.shape", G_.shape, flush=True)
                #        print("rank G_", np.linalg.matrix_rank(G_), flush=True)
                #        print("min eig G_", min(np.linalg.eig(G_)[0]), flush=True)

                b_sol__,L_append_1,L_append_2,B_,A_B_ = bLARS_tournament_2(A_chosen,A_2B,b,int(how_many_3),b_sol[:,0],p,L)


                #print("processor ", p, "A_2B.shape", A_2B.shape, flush=True)
                #print("processor ", p, "A_B_.shape", A_B_.shape, flush=True)

                b_sol__ = b_sol__.reshape(n,1)

                src[n+3,0] = A_B_.shape[1]

                src[0:n+3,0:A_B_.shape[1]] = np.concatenate((A_B_,[proc_v[B_],indices[B_],real_indices[B_]]),axis=0)
                #src[0:n+3,block:block+1] = np.concatenate((b_sol__,[proc_v[B_],indices[B_],real_indices[B_]]),axis=0)
                src[0:n,block:block+1] = b_sol__

                l_L_append_1 = L_append_1.shape[0]*L_append_1.shape[1]
                #print("processor ", p, "l_L_append_1 ", l_L_append_1, flush=True)
                #print("processor ", p, "src.shape ", src.shape, flush=True)
                src[0:l_L_append_1,block+1:block+2] = L_append_1.reshape(l_L_append_1,1)

                l_L_append_2 = L_append_2.shape[0]*L_append_2.shape[1]
                #print("processor ", p, "l_L_append_2 ", l_L_append_2, flush=True)
                #print("processor ", p, "src[0:l_L_append_2,block+2:block+3].shape ", src[0:l_L_append_2,block+2:block+3].shape, flush=True)
                src[0:l_L_append_2,block+2:block+3] = L_append_2.reshape(l_L_append_2,1)
                
                y[:] = src

        op = MPI.Op.Create(my_map, commute=False)

        dim1 = n + 4
        dim2 = block*(A_chosen.shape[1] + block)
        dim = max([dim1,dim2])
        
        dst = np.zeros((dim,block+3))    
        src = np.zeros((dim,block+3))

        if B != []:
            temp = [A_T_rows_ind_local[A_hallow_local_c[ct]] for ct in B]
            src[n+3,0] = A_B.shape[1]   
        else:
            temp = [0]*A_B.shape[1]
            B = [0]*A_B.shape[1]
            src[n+3,0] = 0

        src[0:n+3,0:A_B.shape[1]] = np.concatenate((A_B,[[p]*A_B.shape[1],B,temp]),axis=0)
        src[0:n,block] = b_sol_
        all_srcs.append(src)
        
            # print(src)
        #print("processor ", p, "block ", block, flush=True)
        #print("processor ", p, "A_chosen.shape[1] ", A_chosen.shape[1], flush=True)
        #print("processor ", p, "dst.shape ", dst.shape, flush=True)
        #print("processor ", p, "src.shape ", src.shape, flush=True)
        comm.Allreduce(src, dst, op)

        op.Free()

        if size > 1:
            how_many = int(dst[n+3,0])
        else:
            how_many = A_B.shape[1]

        if A_chosen.shape[1] > 0:
            L = np.concatenate((L, np.zeros((A_chosen.shape[1],how_many))), axis=1)

            if size > 1:
                L_append_1 = dst[0:how_many*A_chosen.shape[1],how_many+1].reshape(how_many,A_chosen.shape[1])
                L_append_2 = dst[0:how_many**2,how_many+2].reshape(how_many,how_many)

            L = np.concatenate((L, np.concatenate((L_append_1, L_append_2), axis=1)), axis=0)
        else:
            if size > 1:
                L = dst[0:how_many**2,how_many+2].reshape(how_many,how_many)
            else:
                L = L_append_2
                

        A_chosen = sp.hstack([A_chosen,dst[0:n,0:how_many]]).tocsr()

        b_sol = dst[0:n,block].reshape(n,1)

        for ct in range(how_many):
            proc = int(dst[n,ct])
            idx = int(dst[n+1,ct])
            idx2 = int(dst[n+2,ct])
            A_hallow_winners.append(idx2)        
            if p == proc:
                A_hallow_local.append(A_hallow_local_c[idx])
                A_hallow.append(A_T_rows_ind_local[idx])

        A_hallow_local_c = diff(range(m_local), A_hallow_local)

        #if p == 0:
        #    print("Processor: ", p, "A_hallow_winners: ",A_hallow_winners, flush=True)
        
        # The code below is just for measuring statistics and its running time is not measured.
        if p == 0:
            end = time.time()
            time_total  = time_total + (end - start)
            time_vector.extend([time_total])
            
        cc = np.zeros(m)
        cc_local = A_local.T.dot((b - b_sol.T).T)
        
        comm.Gatherv(cc_local, cc)

        if p == 0:

            linfty_result.extend([max(abs(cc))])
            l2_result.extend([np.linalg.norm(cc,2)])

            if len(LARS_Ahallow) != 0:
                if block == 1:
                    temp = []
                    temp.extend(LARS_Ahallow[0:block])
                    if len(diff(temp,A_hallow_winners)) != 0:
                        error_Ahallow.extend([len(diff(temp,A_hallow_winners))])
                    else:
                        error_Ahallow.extend([0])
                else:
                    if len(diff(LARS_Ahallow[0:block],A_hallow_winners)) != 0:
                        error_Ahallow.extend([len(diff(LARS_Ahallow[0:block],A_hallow_winners))])
                    else:
                        error_Ahallow.extend([0])
            start = time.time()

    return A_hallow_winners, error_Ahallow, linfty_result, l2_result, time_vector
#        print("Processor: ", p, "A_hallow_winners: ",A_hallow_winners, flush=True)
#        end = time.time()
#        print("Time: ", end-start)

