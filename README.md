[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

# KinetiX
A software toolkit for generating CPU and GPU code to compute reaction kinetics, thermodynamics, and transport properties from a chemical reaction mechanism.

## kinetix python package 

### Requirements
* Python v3.8 and later


### Local installation (optional)
Using the downloaded source code, kinetix can be installed as a Python module:
```sh
python3 install . 
```

### Usage
#### If locally installed
```sh
python3 -m kinetix --mechanism kinetix/mechanisms/gri30.yaml --output kinetix/out/mechanisms/gri
```

#### Standalone usage
```sh
python3 kinetix/__main__.py --mechanism kinetix/mechanisms/gri30.yaml --output kinetix/out/mechanisms/gri
```

## Benchmark kernels

### Requirements:
#### Minimum
* Linux, Mac OS X (Microsoft Windows is not supported)
* C++17/C99 compatible compilers + GNU/Intel Fortran
* MPI-3.1 or later
* CMake version 3.21 or later
#### Optional
* CUDA 9 or later 
* HIP 4.2 or later
* SYCL 2020 or later
* Cantera 2.5 or later

### Installation:

```sh
cd benchmark; mkdir build; cd build
cmake [-DCMAKE_INSTALL_PREFIX=$NEKRK_PATH] [-DAMReX_GPU_BACKEND=CUDA] [-DAMReX_PRECISION=SINGLE] ..
make -j4 install
```
### BK1: Species Production Rates

```sh
> mpirun -np 1 bin/kinetix_bk --backend CUDA --n-states 1000000 --mode 1 --yaml-file kinetix/mechanisms/gri30.yaml
```

### BK2: Mixture-Averaged Transport Properties

```sh
> mpirun -np 1 bin/kinetix_bk --backend CUDA --n-states 1000000 --mode 2  --yaml-file kinetix/mechanisms/gri30.yaml
```

## Examples

Check the `examples/` directory to see how the generated routines can be integrated in a simple MPI program without OCCA dependency (CPU only).

## License

KinetiX is licensed under the BSD-2 Clause License - see the [LICENSE](LICENSE) file for details.
