# Introduction
Generation of reaction kinetics, thermodynamics and transport properties kernels

## How to build

Requirements:
* Linux, Mac OS X (Microsoft Windows is not supported)
* C++17/C99 compatible compilers + GNU/Intel Fortran
* MPI-3.1 or later
* CMake version 3.21 or later
* Python v3.8 and later
* (CUDA, HIP, SERIAL/C++)

Installation:

```sh
mkdir build; cd build
cmake [-DCMAKE_INSTALL_PREFIX=$NEKRK_PATH] [-DAMReX_GPU_BACKEND=CUDA] ..
make -j4 install
```

## Benchmark Kernels

### BK1: Production Rates

```sh
> mpirun -np 1 bin/nekrk_bk --backend CUDA --n-states 1000000 --mode 1 --yaml-file mechanisms/gri30.yaml
```

### BK2: Mixture-Averaged Transport Properties

```sh
> mpirun -np 1 bin/nekrk_bk --backend CUDA --n-states 1000000 --mode 2  --yaml-file mechanisms/gri30.yaml
```
