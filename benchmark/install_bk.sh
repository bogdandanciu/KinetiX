#!/bin/bash

# Set default values
KINETIX_PATH=${KINETIX_PATH:-$HOME/.local/kinetix}
AMREX_GPU_BACKEND=${AMREX_GPU_BACKEND:-}
AMREX_PRECISION=${AMREX_PRECISION:-}

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --kinetix-path PATH          Set custom installation path (default: $HOME/.local/kinetix)"
    echo "  --amrex-gpu-backend BACKEND  Set AMReX GPU backend (e.g., CUDA) (default:NONE)"
    echo "  --amrex-precision PRECISION  Set AMReX precision (e.g., SINGLE) (default:DOUBLE)"
    echo "  -h, --help                   Display this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --kinetix-path)
            KINETIX_PATH="$2"
            shift 2
            ;;
        --gpu-backend)
            AMREX_GPU_BACKEND="$2"
            shift 2
            ;;
        --precision)
            AMREX_PRECISION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

export KINETIX_PATH

# Prepare CMake arguments
CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$KINETIX_PATH"
if [ -n "$AMREX_GPU_BACKEND" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DAMReX_GPU_BACKEND=$AMREX_GPU_BACKEND"
fi
if [ -n "$AMREX_PRECISION" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DAMReX_PRECISION=$AMREX_PRECISION"
fi

# Run the installation process
set -e  # Exit on any error
mkdir -p build
cd build
cmake $CMAKE_ARGS ..
make -j4 install
cd ../../

echo "BK has been successfully installed to $KINETIX_PATH"
