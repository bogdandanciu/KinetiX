# Introduction
Generation of thermochemical and transport properties

## How to build
Dependencies:
- Python v3.8 and later

```sh
mkdir build; cd build
cmake [-DCMAKE_INSTALL_PREFIX=$NEKRK_PATH] ..
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
