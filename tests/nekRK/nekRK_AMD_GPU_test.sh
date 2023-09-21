#!/bin/bash
#SBATCH -A project_465000567 
#SBATCH -J nekRK
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p standard-g
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2

# -------
# 2560 GPU compute nodes
# 4× NVIDIA MI250x GPUs
# -------

ulimit -s unlimited
##############
#gri30-20 - 5M 
srun bin/nekcrf_bk --backend HIP --n-states 238000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 238000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 238000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 238000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

#gri30-20 - 20M
srun bin/nekcrf_bk --backend HIP --n-states 952000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 952000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 952000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 952000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

#gri30-20 - 40M
srun bin/nekcrf_bk --backend HIP --n-states 1904000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 1904000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 1904000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend HIP --n-states 1904000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

