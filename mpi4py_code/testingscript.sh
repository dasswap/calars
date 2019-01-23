#!/bin/bash
# Script to run BLARs

echo Compiling Code Regular
mpic++ mpi4py.cpp -o main -O2 -DARMA_DONT_USE_WRAPPER -DARMA_USE_HDF5 -DARMA_USE_LAPACK -DARMA_USE_BLAS -llapack -lblas -lhdf5_serial -std=c++14

echo Compiling Code Tournament
mpic++ mpi4py_tourney.cpp -o main_t -O2 -DARMA_DONT_USE_WRAPPER -DARMA_USE_HDF5 -DARMA_USE_LAPACK -DARMA_USE_BLAS -llapack -lblas -lhdf5_serial -std=c++14

# files='abalone_scale.txt w4a.txt cadata.txt cpusmall_scale.txt housing_scale.txt mg_scale.txt mpg_scale.txt pyrim_scale.txt space_ga_scale.txt triazines_scale.txt'

read -p 'Name of file: ' file
read -p 'Number of threads: ' threads

echo $file
# Python wrapper
echo Running Python Wrapper
python mpi4pyTime.py $threads $file

#Compiler for Jakestown

#Run MPI
counter=1
until [ $counter -gt 10 ]
do
	echo Test Number $counter

	echo Running MPI Code Regular C++
	mpirun -np $threads main 

	echo Running MPI Code Tournament C++ 
	mpirun -np $threads main_t 

	echo Running MPI Code Regular Python
	mpirun -np $threads python generate_results.py $file

	echo Running MPI Code Tournament Python
	mpirun -np $threads python generate_results_tournament.py $file

	((counter++))
done

echo Tests completed.
