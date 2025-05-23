#!/usr/bin/env bash
set -e

cd ..
mkdir -p build-cmake
cd build-cmake

TORCH_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')

cmake ../csrc \
    -GNinja \
    -DCMAKE_PREFIX_PATH=$TORCH_PREFIX_PATH \
    -DTORCH_CUDA_ARCH_LIST=9.0a+PTX \
    -DWITH_TESTS=ON \
    -DWITH_BENCHMARKS=ON

ninja test_all_to_all bench_all_to_all
