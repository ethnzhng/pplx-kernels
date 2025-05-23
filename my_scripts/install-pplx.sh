#!/usr/bin/env bash
set -e

export NVSHMEM_DIR=/usr/local/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd ..
TORCH_CUDA_ARCH_LIST=9.0a+PTX python3 setup.py bdist_wheel
pip install dist/*.whl

# pip install --force-reinstall --no-deps dist/*.whl
