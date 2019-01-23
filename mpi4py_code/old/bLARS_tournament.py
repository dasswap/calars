#passRandomDraw.py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_linalg

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

def bLARS_tournament(A,b,m,n,block,A_hallow,b_sol,processor):

    r = b - b_sol
    c = A.T.dot(r)
    l_A_hallow = len(A_hallow)
    B = np.argsort(-abs(c))[0:block]
    c_const = abs(c[B[block-1]])
    
    #print("Processor: ",processor,"c_const: ",c_const)
    
    if A_hallow == []:
        A_hallow.extend(B)
        
        return (b_sol,A_hallow,B)

    A_hallow_old = list(A_hallow)
    l_A_hallow = len(A_hallow)
    l_A_hallow_old = len(A_hallow_old)
    A_hallow_c = diff(range(m), A_hallow)   
    
    G = A[:,A_hallow].T.dot(A[:,A_hallow]).toarray()

    w = np.zeros(l_A_hallow)
    h = np.zeros(1)

    #print("A_hallow: ",A_hallow)
    L = np.linalg.cholesky(G)
    #print("L: ", L)
    s = c[A_hallow] 
    #print("s: ",s)
    q = solve_linear(L,s)
    #print("q: ",q)
    h = np.asarray([1/np.sqrt(np.dot(s.T,q))])
    w = np.multiply(h,q)
    #print("w: ",w)

    u = A[:,A_hallow].dot(w)
    
    #print("norm(u): ", np.linalg.norm(u))

    a = A.T.dot(u)
    
    #print("norm(a): ", np.linalg.norm(a))
    #print("h: ", h) 

    l_A_hallow_c = len(A_hallow_c)
    
    if l_A_hallow_c == 0:
        return (b_sol,A_hallow,[])

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

    #print("gamma: ", gamma)   
        
    b_sol = b_sol + np.multiply(gamma,u)

    A_hallow.extend(B)

    return (b_sol,A_hallow,B)
