#!/bin/bash -ex

for dataset in $(cat datasets.txt); do
    ntasks=1
    value=`cat ./n_powersOf2/$dataset`
    echo $value
    while [ $ntasks -le 4 ]
    do
        export dataset=$dataset
        mpiexec -np $ntasks python3 generate_results.py
        ntasks=$(( $ntasks * 2 ))
    done
done
