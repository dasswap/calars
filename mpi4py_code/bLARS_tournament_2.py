#passRandomDraw.py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_linalg

# Just a useful function.
def diff(a, b):
		b = set(b)
		return [aa for aa in a if aa not in b]
	
def solve_linear(L,b):
	ide = np.eye(L.shape[0], L.shape[1])
	L = L + (ide/2)
	y = sp_linalg.solve_triangular(L,b,lower=True)
	return sp_linalg.solve_triangular(L,y,trans='T',lower=True)

def solve_quadratic(rhs):
	n = rhs.shape[0]
	Omega = np.zeros((n,n))
	ide = np.eye(n, rhs.shape[1])
	rhs = rhs + (ide/2)

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

def stepLARS(c_const, h, c, a, idx):
	if (c_const > abs(idx)):
		if (np.sign(c[idx]) == np.sign(a[idx])):
			val1 = (c_const - c[idx])/(h*c_const - a[idx])
			val2 = (c_const + c[idx])/(h*c_const + a[idx])
			to_ret = min(val for val in [val1,val2] if val > 0)
		else:
			to_ret = (c_const - c[idx])/(h*c_const + a[idx])
	else:
		if (np.sign(c[idx]) == np.sign(a[idx])):
			if ((c[idx]*h) >= (c_const*a[idx])):
				to_ret = (c_const + c[idx])/(h*c_const + a[idx])
			else:
				to_ret = 1/h
		else:
			to_ret = 0
	return to_ret


def bLARS_tournament_2(A,A_c,b,block,b_sol,processor,L):

	l_A_c = A_c.shape[1]
	if l_A_c == 0:
		return (b_sol,[],[],[],np.zeros((b.shape[0],block)))
	
	r = b - b_sol
	
	l_A = A.shape[1]
	
	A_full = sp.hstack([A,A_c]).tocsr()
	
	G_ = A_full.T.dot(A_full).toarray();
	
	#print("G_.shape", G_.shape, flush=True)
	#print("rank G_", np.linalg.matrix_rank(G_), flush=True)
	#print("min eig G_", min(np.linalg.eig(G_)[0]), flush=True)
	if l_A == 0:
		c = A_c.T.dot(r)
		#B = np.argsort(-abs(c))[0] + l_A
		B = np.argsort(-abs(c))[0]
		
		#G = (A_full[:,B].T.dot(A_full[:,B])).toarray()
		
		#L = np.linalg.cholesky(G)
		
		#return (b_sol,L,np.zeros((0,0)),list(B),A_full[:,B].toarray())
		
		A_hallow = list([B])
		iterations = min(block - 1,l_A_c)
	else:
		A_hallow = list(range(l_A))
		iterations = min(block,l_A_c)
		
	A_hallow_c = diff(range(l_A + l_A_c), A_hallow)
	A_hallow_old = list(A_hallow)
	
	l_A_hallow = len(A_hallow)
	l_A_hallow_c = len(A_hallow_c)
	l_A_hallow_old = len(A_hallow_old)
	
	c = A_full.T.dot(r)
	
	c_const = np.amax(abs(c))
 
	if l_A_hallow == 1:
		G = (A_full[:,A_hallow].T.dot(A_full[:,A_hallow])).toarray()
		L = np.linalg.cholesky(G)

	for iteration in range(iterations):
			
		
		#print("b_sol: ", b_sol, flush=True)

		w = np.zeros(l_A)

		s = c[A_hallow] 
		
		
		q = solve_linear(L,s)
		
		#print("q: ", q, flush=True)
		
		h = np.asarray([1/np.sqrt(np.dot(s.T,q))])
		
		#print("h: ", h, flush=True)
		
		w = np.multiply(h,q)
		
		#print("w: ", w, flush=True)

		u = A_full[:,A_hallow].dot(w)
		
		a = A_full.T.dot(u)
		
		#print("a: ", a, flush=True)
		min_pos = np.zeros(l_A_hallow_c)
		for j in range(l_A_hallow_c):
			idx = A_hallow_c[j]
			min_val = stepLARS(c_const, h, c, a, idx)
			min_pos[j] = min_val

		idx = np.argsort(min_pos)[0].tolist()
		B = np.asarray(A_hallow_c)[idx].tolist()
		gamma = 0.9*min_pos[idx] # I decrease the original stepsize because of numerical issues.
		
		#print("gamma: ", gamma, flush=True)

		b_sol = b_sol + np.multiply(gamma,u)
		
		c = c - gamma*a
		c_const = np.amax(abs(c))
		
		A_hallow.extend([B])
		l_A_hallow = len(A_hallow)
		A_hallow_c = diff(range(l_A + l_A_c), A_hallow)
		l_A_hallow_c = len(A_hallow_c)

		A_full_hallow_t_old_A_B = (A_full[:,A_hallow_old].T.dot(A_full[:,B])).toarray()

		A_full_B_t_A_full_B = (A_full[:,B].T.dot(A_full[:,B])).toarray()  

		ide = np.eye(L.shape[0], L.shape[1])
		L = L + (ide/2)

		H = sp_linalg.solve_triangular(L,A_full_hallow_t_old_A_B,lower=True)

		rhs_Omega = A_full_B_t_A_full_B - np.dot(H.T,H)

		Omega = solve_quadratic(rhs_Omega)

		temp4 = np.concatenate((H.T, Omega), axis=1)
		L = np.concatenate((L, np.zeros((l_A_hallow_old,1))), axis=1)
		L = np.concatenate((L, temp4), axis=0)
		
		#G = (A_full[:,A_hallow].T.dot(A_full[:,A_hallow])).toarray()
		#print("G.shape", G.shape, flush=True)
		#print("rank G", np.linalg.matrix_rank(G), flush=True)
		#print("min eig G", min(np.linalg.eig(G)[0]), flush=True)
		#print("len(A_hallow) ", len(A_hallow))
		#print("len(set(A_hallow)) ", len(set(A_hallow)))
		#L = np.linalg.cholesky(G)
		#print("rank L", np.linalg.matrix_rank(L), flush=True)
		#print("min eig L", min(np.linalg.eig(L)[0]), flush=True)
		
		#print("L: ", L)
		A_hallow_old = list(A_hallow)
		l_A_hallow_old = len(A_hallow_old)

	L_append_1 = L[l_A:l_A+block,0:l_A]    
	L_append_2 = L[l_A:l_A+block,l_A:l_A+block]
		
	which_col = list(np.asarray(A_hallow[l_A:l_A+block]) - l_A)
	   
	return (b_sol,L_append_1,L_append_2,which_col,A_full[:,A_hallow[l_A:l_A+block]].toarray())
	