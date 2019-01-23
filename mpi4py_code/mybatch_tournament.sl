#!/bin/bash -l

#SBATCH -c 1
#SBATCH -t 02:00:00
#SBATCH -p regular

module load python/3.6-anaconda-4.4

echo running on $dataset

srun -n $SLURM_NTASKS -c 1 python generate_results_tournament.py
