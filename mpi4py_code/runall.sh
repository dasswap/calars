#!/bin/bash -ex

for dataset in $(cat datasets.txt); do
    ntasks=1
    value=`cat ./n_powersOf2/$dataset`
    while [ $ntasks -le $value ]
    do
        export dataset=$dataset
        sbatch -n $ntasks -c 1 --export=dataset=$dataset mybatch.sl
        ntasks=$(( $ntasks * 2 ))
    done
done

