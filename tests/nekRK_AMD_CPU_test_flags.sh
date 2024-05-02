#!/bin/bash -x
#SBATCH --job-name=nekRK_CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=standard
#SBATCH --account=project_465001076
#SBATCH --time=02:00:00
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
# -------
# 1536 CPU compute nodes
# 2× AMD EPYC 7763 CPUs (128 cores)
# -------

#module purge
#module load PrgEnv-gnu
#module load gcc cray-python rocm/5.2.3
#module load craype cray-mpich
module list

ulimit -s unlimited
##############

#LiDryer
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --group-vis
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --fit-rcpDiffCoeffs
rm -rf .cache
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 1 --fit-rcpDiffCoeffs --nonsymDij
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele

#gri30 
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --group-vis
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --fit-rcpDiffCoeffs
rm -rf .cache
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 1 --fit-rcpDiffCoeffs --nonsymDij
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele

#Konnov 
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --loop-gibbsexp --group-rxnUnroll
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --group-vis
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --fit-rcpDiffCoeffs
rm -rf .cache
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 1 --fit-rcpDiffCoeffs --nonsymDij
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 100 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele

