#!/bin/bash -x
#SBATCH --job-name=nekRK_GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --account=dems
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --distribution=block:block:fcyclic

# -------
# 936 GPU compute nodes
# 4× NVIDIA A100 GPUs
# -------

ulimit -s unlimited
##############
#gri30-20 - 5M 
srun bin/nekcrf_bk --backend CUDA --n-states 238000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 238000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 238000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 238000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

#gri30-20 - 20M
srun bin/nekcrf_bk --backend CUDA --n-states 952000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 952000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 952000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 952000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

#gri30-20 - 40M
srun bin/nekcrf_bk --backend CUDA --n-states 1904000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1904000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1904000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1904000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

