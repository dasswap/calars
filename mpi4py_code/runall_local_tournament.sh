#!/bin/bash -ex

for dataset in $(cat datasets.txt); do
    ntasks=32
    value=`cat ./m_powersOf2/$dataset`
    while [ $ntasks -le 32 ]
    do
        export dataset=$dataset
        mpiexec -l -np $ntasks python3 -m cProfile generate_results_tournament.py >> profile_results.txt
        ntasks=$(( $ntasks * 2 ))
    done
done

