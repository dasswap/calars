
# File to run tests.

import os
import sys

def range_one(x):
	return list(range(x+1))[1:]

def avg(arr):
	return sum(arr)/len(arr)



# done until thread = 4, block = 3
def test_stuff():
	for thread in range_one(10):
		# running the cpp preparation code for a given thread count
		if (thread >= 4):
			os.system("python mpi4pyTime.py " + str(thread))
			for block in range_one(4):
				if (thread == 4 and block == 4) or (thread > 4):
					print("We are using block size " + str(block) + ".\n")
					for trial in range_one(10):
						print("This is trial number " + str(trial) + ".\n")
						# running the python code 
						os.system("mpirun -np " + str(thread) + " python generate_results.py " + str(block))
						# running the c code
						os.system("mpirun -np " + str(thread) + " main " + str(block))

def return_values():
	for thread in range_one(10):
		for block in range_one(4):
			# python values:
			f = open("pb" + str(block) + "t" + str(thread), "r")
			data = [float(x) for x in f.readlines()]
			print("Python: block " + str(block) + " and # threads " + str(thread) + ": " + str(avg(data)) + "\n")
			f = open("cb" + str(block) + "t" + str(thread), "r")
			data = [float(x) for x in f.readlines()]
			print("C++: block " + str(block) + " and # threads " + str(thread) + ": " + str(avg(data)) + "\n")

test_stuff()
return_values()