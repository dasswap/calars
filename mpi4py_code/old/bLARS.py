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

def bLARS(A,b,m,n,block):

    iterations = int(np.floor(m/block))

    b_sol = np.zeros(n)
    r = np.asarray(b)

    c = A.T.dot(r)

    A_hallow = []
    B = np.argsort(-abs(c))[0:block]
    c_const = abs(c[B[block-1]])
    A_hallow.extend(B)
    A_hallow_old = list(A_hallow)
    l_A_hallow = len(A_hallow)
    l_A_hallow_old = len(A_hallow_old)
    A_hallow_c = diff(range(m), A_hallow)

    G = A[:,A_hallow].T.dot(A[:,A_hallow]).toarray()

    w = np.zeros(l_A_hallow)
    h = np.zeros(1)

    L = np.linalg.cholesky(G)
    s = c[A_hallow] 
    q = solve_linear(L,s)
    h = np.asarray([1/np.sqrt(np.dot(s.T,q))])
    w = h*q

    u = A[:,A_hallow].dot(w)

    a = A.T.dot(u)

    print("c_const: ",c_const)

    for i in range(iterations-1):

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

        b_sol = b_sol + np.multiply(gamma,u)

        c[A_hallow] = c[A_hallow]*(1-gamma*h)
        c[A_hallow_c] = c[A_hallow_c] - gamma*a[A_hallow_c]
        c_const = c_const*(1-gamma*h)

        A_hallow.extend(B)
        l_A_hallow = len(A_hallow)
        A_hallow_c = diff(range(m), A_hallow)
        l_B = len(B)

        A_hallow_t_old_A_B = A[:,A_hallow_old].T.dot(A[:,B]).toarray()

        A_B_t_A_B = A[:,B].T.dot(A[:,B]).toarray()  

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

        u = A[:,A_hallow].dot(w)

        a = A.T.dot(u)

        A_hallow_old = list(A_hallow)
        l_A_hallow = len(A_hallow)
        l_A_hallow_old = len(A_hallow_old)

        print("c_const: ",c_const)

    print("A_hallow: ",A_hallow)
