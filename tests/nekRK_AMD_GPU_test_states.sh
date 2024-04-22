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

###LiDryer
# 2M 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 5M 
srun ../bin/nekrk_bk --backend HIP --n-states 500000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 500000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 10M 
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 20M 
srun ../bin/nekrk_bk --backend HIP --n-states 2000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 2000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 50M 
srun ../bin/nekrk_bk --backend HIP --n-states 5000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 5000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 100M 
srun ../bin/nekrk_bk --backend HIP --n-states 10000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 10000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 200M 
srun ../bin/nekrk_bk --backend HIP --n-states 20000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 20000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele

###gri30 
#5M
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 92590 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 92590 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
#10M
srun ../bin/nekrk_bk --backend HIP --n-states 185180 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 185180 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
#20M
srun ../bin/nekrk_bk --backend HIP --n-states 370360 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 370360 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
#50M
srun ../bin/nekrk_bk --backend HIP --n-states 925900 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 925900 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
#100M
srun ../bin/nekrk_bk --backend HIP --n-states 1851800 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 1851800 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
#200M
srun ../bin/nekrk_bk --backend HIP --n-states 3703600 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 3703600 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele

###Konnov 
#2M
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 15384 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 15384 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
#5M
srun ../bin/nekrk_bk --backend HIP --n-states 38460 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 38460 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
#10M
srun ../bin/nekrk_bk --backend HIP --n-states 76920 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 76920 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
#20M
srun ../bin/nekrk_bk --backend HIP --n-states 153840 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 153840 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
#50M
srun ../bin/nekrk_bk --backend HIP --n-states 384600 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 384600 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
#100M
srun ../bin/nekrk_bk --backend HIP --n-states 769200 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 769200 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
#200M
srun ../bin/nekrk_bk --backend HIP --n-states 1538400 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --fit-rcpDiffCoeffs
srun ../bin/nekrk_bk --backend HIP --n-states 1538400 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele

