#!/bin/bash -x
#SBATCH --job-name=nekRK_GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --account=dems
#SBATCH --time=08:00:00
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

###LiDryer
# 50K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 5000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 5000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 100K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 10000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 10000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 200K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 20000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 20000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 500K
srun ../bin/nekrk_bk --backend CUDA --n-states 50000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 50000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 1M
srun ../bin/nekrk_bk --backend CUDA --n-states 100000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 100000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 2M 
srun ../bin/nekrk_bk --backend CUDA --n-states 200000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 200000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 5M 
srun ../bin/nekrk_bk --backend CUDA --n-states 500000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 500000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 10M 
srun ../bin/nekrk_bk --backend CUDA --n-states 1000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 1000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 20M 
srun ../bin/nekrk_bk --backend CUDA --n-states 2000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 2000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 50M 
srun ../bin/nekrk_bk --backend CUDA --n-states 5000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 5000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 100M 
srun ../bin/nekrk_bk --backend CUDA --n-states 10000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 10000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 200M 
srun ../bin/nekrk_bk --backend CUDA --n-states 20000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 20000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 500M 
srun ../bin/nekrk_bk --backend CUDA --n-states 50000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 50000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 1B 
srun ../bin/nekrk_bk --backend CUDA --n-states 100000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 100000000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele

###gri30 
# 50K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 925 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 925 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 100K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 1851 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 1851 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 200K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 3703 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 3703 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 500K
srun ../bin/nekrk_bk --backend CUDA --n-states 9259 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 9259 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 1M
srun ../bin/nekrk_bk --backend CUDA --n-states 18518 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 18518 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 2M
srun ../bin/nekrk_bk --backend CUDA --n-states 37037 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 37037 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 5M
srun ../bin/nekrk_bk --backend CUDA --n-states 92590 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 92590 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 10M
srun ../bin/nekrk_bk --backend CUDA --n-states 185180 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 185180 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 20M
srun ../bin/nekrk_bk --backend CUDA --n-states 370360 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 370360 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 50M
srun ../bin/nekrk_bk --backend CUDA --n-states 925900 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 925900 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 100M
srun ../bin/nekrk_bk --backend CUDA --n-states 1851800 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 1851800 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 200M
srun ../bin/nekrk_bk --backend CUDA --n-states 3703600 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 3703600 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 500M
srun ../bin/nekrk_bk --backend CUDA --n-states 9259000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 9259000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 1B
srun ../bin/nekrk_bk --backend CUDA --n-states 18518000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend CUDA --n-states 18518000 --n-repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele

###Konnov 
# 50K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 384 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 384 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 100K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 769 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 769 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 200K
rm -rf .cache/
srun ../bin/nekrk_bk --backend CUDA --n-states 1538 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 1538 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 500K
srun ../bin/nekrk_bk --backend CUDA --n-states 3846 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 3856 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 1M
srun ../bin/nekrk_bk --backend CUDA --n-states 7692 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 7692 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 2M
srun ../bin/nekrk_bk --backend CUDA --n-states 15384 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 15384 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 5M
srun ../bin/nekrk_bk --backend CUDA --n-states 38460 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 38460 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 10M
srun ../bin/nekrk_bk --backend CUDA --n-states 76920 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 76920 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 20M
srun ../bin/nekrk_bk --backend CUDA --n-states 153840 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 153840 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 50M
srun ../bin/nekrk_bk --backend CUDA --n-states 384600 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 384600 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 100M
srun ../bin/nekrk_bk --backend CUDA --n-states 769200 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 769200 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 200M
srun ../bin/nekrk_bk --backend CUDA --n-states 1538400 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 1538400 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 500M
srun ../bin/nekrk_bk --backend CUDA --n-states 3846000 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 3846000 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 1B
srun ../bin/nekrk_bk --backend CUDA --n-states 7692000 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops --loop-gibbsexp --fit-rcpDiffCoeffs --group-vis
srun ../bin/nekrk_bk --backend CUDA --n-states 7692000 --n-repetitions 40 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele

