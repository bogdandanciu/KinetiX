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

###LiDryer
rm -rf .cache/
# 200K 
srun ../bin/nekrk_bk --backend SERIAL --n-states 20000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 20000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 500K 
srun ../bin/nekrk_bk --backend SERIAL --n-states 50000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 50000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 1M
srun ../bin/nekrk_bk --backend SERIAL --n-states 100000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 100000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 2M 
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 200000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 5M 
srun ../bin/nekrk_bk --backend SERIAL --n-states 500000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 500000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 10M 
srun ../bin/nekrk_bk --backend SERIAL --n-states 1000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 1000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 20M 
srun ../bin/nekrk_bk --backend SERIAL --n-states 2000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 2000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 50M 
srun ../bin/nekrk_bk --backend SERIAL --n-states 5000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 5000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 100M 
srun ../bin/nekrk_bk --backend SERIAL --n-states 10000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 10000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 200M 
srun ../bin/nekrk_bk --backend SERIAL --n-states 20000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 20000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 500M 
srun ../bin/nekrk_bk --backend SERIAL --n-states 50000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 50000000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele
# 1B 
srun ../bin/nekrk_bk --backend SERIAL --n-states 100000000 --repetitions 10 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 100000000 --repetitions 10 --mode 0 --yaml-file ../mechanisms/LiDryer.yaml --tool Pele


###gri30 
# 200K
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 3703 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 3703 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 500K 
srun ../bin/nekrk_bk --backend SERIAL --n-states 9259 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 9259 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 1M
srun ../bin/nekrk_bk --backend SERIAL --n-states 18519 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 18519 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 2M
srun ../bin/nekrk_bk --backend SERIAL --n-states 37038 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 37038 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 5M
srun ../bin/nekrk_bk --backend SERIAL --n-states 92590 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 92590 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 10M
srun ../bin/nekrk_bk --backend SERIAL --n-states 185180 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 185180 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 20M
srun ../bin/nekrk_bk --backend SERIAL --n-states 370360 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 370360 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 50M
srun ../bin/nekrk_bk --backend SERIAL --n-states 925900 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 925900 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 100M
srun ../bin/nekrk_bk --backend SERIAL --n-states 1851800 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 1851800 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 200M
srun ../bin/nekrk_bk --backend SERIAL --n-states 3703600 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 3703600 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 500M
srun ../bin/nekrk_bk --backend SERIAL --n-states 9259000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 9259000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 9259000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele
# 1B
srun ../bin/nekrk_bk --backend SERIAL --n-states 18518000 --repetitions 10 --mode 0 --yaml-file ../mechanisms/gri30.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 18518000 --repetitions 10 --mode 0 --yaml-file ../mechanisms/gri30.yaml --tool Pele

###Konnov 
# 200K
rm -rf .cache/
srun ../bin/nekrk_bk --backend SERIAL --n-states 1538 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 1538 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 500K
srun ../bin/nekrk_bk --backend SERIAL --n-states 3846 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 3846 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 1M
srun ../bin/nekrk_bk --backend SERIAL --n-states 7692 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 7692 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 2M
srun ../bin/nekrk_bk --backend SERIAL --n-states 15384 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 15384 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 5M
srun ../bin/nekrk_bk --backend SERIAL --n-states 38460 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 38460 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 10M
srun ../bin/nekrk_bk --backend SERIAL --n-states 76920 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 76920 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 20M
srun ../bin/nekrk_bk --backend SERIAL --n-states 153840 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 153840 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 50M
srun ../bin/nekrk_bk --backend SERIAL --n-states 384600 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 384600 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 100M
srun ../bin/nekrk_bk --backend SERIAL --n-states 769200 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 769200 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 200M
srun ../bin/nekrk_bk --backend SERIAL --n-states 1538400 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 1538400 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 500M
srun ../bin/nekrk_bk --backend SERIAL --n-states 3846000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 3846000 --repetitions 30 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele
# 1B
srun ../bin/nekrk_bk --backend SERIAL --n-states 7692000 --repetitions 10 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --unroll-loops 0 --fit-rcpDiffCoeffs --nonsymDij
srun ../bin/nekrk_bk --backend SERIAL --n-states 7692000 --repetitions 10 --mode 0 --yaml-file ../mechanisms/Konnov.yaml --tool Pele

