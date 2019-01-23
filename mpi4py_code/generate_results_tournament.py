import os
from mpi4py import MPI
import numpy as np
import sys
from mpi4py_bLARS_svm import parallel_bLARS_svm_data
import mpi4py_tournament_bLARS_svm_clever
reload(mpi4py_tournament_bLARS_svm_clever)

comm = MPI.COMM_WORLD
size = comm.Get_size()
p = comm.Get_rank()

# ff = os.environ["dataset"]
ff = sys.argv[1]
# ff = 'w4a.txt'

def write_file(block, text, isArr, item):
	f = open("ptourt" + str(size) + "b" + str(block) + text, "a")
	to_write = ""
	if isArr:
		for elem in item:
			to_write += str(elem) + " "	
	else:
		to_write = str(item)
	f.write(to_write + "\n")
	f.close()


m = np.load('./m_data/'+'m_'+ff+'.npz')['m']
n = np.load('./n_data/'+'n_'+ff+'.npz')['n']

max_block = int(np.floor(min(m,n)/2))
LARS_Ahallow, error_Ahallow, linfty_result, l2_result, time_vector = mpi4py_tournament_bLARS_svm_clever.tourney([], 1, ff)

if (p == 0):
	write_file(1, "error", True, error_Ahallow)
	write_file(1, "linfty", True, linfty_result)
	write_file(1, "l2", True, l2_result)
	write_file(1, "time", False, sum(time_vector))
		
for block in range(2,max_block+1):
	
	if p == 0:
		print("block: ",block)
	
	LARS_Ahallow_loop_tour, error_Ahallow_tour, linfty_result_tour, l2_result_tour, time_vector_tour = mpi4py_tournament_bLARS_svm_clever.tourney(LARS_Ahallow,block,ff)
	
	if p == 0:
		write_file(block, "error", True, error_Ahallow_tour)
		write_file(block, "linfty", True, linfty_result_tour)
		write_file(block, "l2", True, l2_result_tour)
		write_file(block, "time", False, sum(time_vector_tour))