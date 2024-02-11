#!/bin/bash -x
#SBATCH --job-name=nekRK_GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --account=dems
#SBATCH --time=04:00:00
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
#gri30 - 53M 
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 0 --loop-gibbsexp
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 0 --group-rxn-repArrh
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 0 --loop-gibbsexp --group-rxn-repArrh
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 0 --nonsymDij
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 0 --nonsymDij --fit-rcpDiffCoeffs
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 1 --nonsymDij
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 1 --fit-rcpDiffCoeffs
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/gri30.yaml --unroll-loops 1 --nonsymDij --fit-rcpDiffCoeffs

#LiDryer -10 M
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 0
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 0 --loop-gibbsexp
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 0 --group-rxn-repArrh
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 0 --loop-gibbsexp --group-rxn-repArrh
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 0 --nonsymDij
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 0 --nonsymDij --fit-rcpDiffCoeffs
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 1
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 1 --nonsymDij
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 1 --fit-rcpDiffCoeffs
rm -rf .cache/
srun bin/nekcrf_bk --backend CUDA --n-states 1000000 --repetitions 100 --mode 0 --yaml-file mechanisms/LiDryer.yaml --unroll-loops 1 --nonsymDij --fit-rcpDiffCoeffs
