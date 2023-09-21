#!/bin/bash -x
#SBATCH --job-name=nekRK_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=standard
#SBATCH --account=project_465000567
#SBATCH --time=02:00:00
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j

# -------
# 1536 CPU compute nodes
# 2× AMD EPYC 7763 CPUs (128 cores)
# -------

ulimit -s unlimited
##############
#gri30-20 - 5M 
srun bin/nekcrf_bk --backend SERIAL --n-states 238000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 238000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 238000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 238000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

#gri30-20 - 20M
srun bin/nekcrf_bk --backend SERIAL --n-states 952000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 952000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 952000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 952000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

#gri30-20 - 40M
srun bin/nekcrf_bk --backend SERIAL --n-states 1904000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 1904000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 1904000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 0 --block-size 256
rm -rf .cache/
srun bin/nekcrf_bk --backend SERIAL --n-states 1904000 --repetitions 10 --mode 0 --yaml-file mechanisms/gri30-20.yaml --unroll-loops 1 --block-size 256
rm -rf .cache/

