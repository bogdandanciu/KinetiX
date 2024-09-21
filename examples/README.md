## Requirements

### Minimum
* Linux, Mac OS X (Microsoft Windows is not supported)
* C++17/C99 compatible compilers + GNU/Intel Fortran
* MPI-3.1 or later
* GNU Make
### Optional
* [AMReX](https://github.com/AMReX-Codes/amrex)
* [Cantera](https://github.com/Cantera/cantera)
* [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)

## Kinetix
Compile 
```sh
make
```
Run 
```sh
mpirun -np 1 ./kinetix_test --n-states 1 --n-repetitions 1 --print-states
```

## PelePhysics
Requires installing the [AMReX](https://github.com/AMReX-Codes/amrex) library. 
The installation directory is assumed to be the $HOME directory in the Makefile.

If the benchmark kernels are installed, AMReX will be included in the installation directory. 
However, the include and library paths in the Makefile must be updated to correctly reference the installed AMReX components.

Compile 
```sh
make
```
Run 
```sh
mpirun -np 1 ./pele_test --n-states 1 --n-repetitions 1 --print-states
```

## Cantera
Requires compiling [Cantera](https://cantera.org/install/compiling-install.html#sec-compiling) from source. 
The pkg-config program is used in the Makefile to determine the correct compiler and linker flags for use with Cantera.

Compile 
```sh
make
```
Run 
```sh
mpirun -np 1 ./cantera_test --n-states 1 --n-repetitions 1 --mechanism ../../kinetix/mechanisms/gri30.yaml --print-states
```