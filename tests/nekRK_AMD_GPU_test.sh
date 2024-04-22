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

#LiDryer
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --group-vis
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --fit-rcpDiffCoeffs
rm -rf .cache
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --fit-rcpDiffCoeffs --nonsymDij 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele

#gri30 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --group-vis
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --fit-rcpDiffCoeffs
rm -rf .cache
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --fit-rcpDiffCoeffs --nonsymDij 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele

#Konnov 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --group-vis
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --fit-rcpDiffCoeffs
rm -rf .cache
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --fit-rcpDiffCoeffs --nonsymDij 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij 
rm -rf .cache/
srun ../bin/nekrk_bk --backend HIP --n-states 1000000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
