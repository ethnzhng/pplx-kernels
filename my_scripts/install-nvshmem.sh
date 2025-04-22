#!/usr/bin/env bash
set -e

# build & install NVSHMEM
cd ~
if [ ! -d "nvshmem_src" ]; then
    wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
    tar -xvf nvshmem_src_3.2.5-1.txz
else
    echo "Directory nvshmem_src already exists, skipping download and extract"
fi
cd nvshmem_src

export CUDA_HOME=/usr/local/cuda-12.2
export LIBFABRIC_HOME=/opt/amazon/efa
export GDRCOPY_HOME=/usr/lib/x86_64-linux-gnu
export NCCL_HOME=/opt/aws-ofi-nccl
export MPI_HOME=/opt/amazon/openmpi

export NVSHMEM_DIR=/usr/local/nvshmem

cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=$NVSHMEM_DIR \
    -DCUDA_HOME=$CUDA_HOME \
    -DLIBFABRIC_HOME=$LIBFABRIC_HOME \
    -DGDRCOPY_HOME=$GDRCOPY_HOME \
    -DNCCL_HOME=$NCCL_HOME \
    -DMPI_HOME=$MPI_HOME \
    -DNVSHMEM_SHMEM_SUPPORT=OFF \
    -DNVSHMEM_UCX_SUPPORT=OFF \
    -DNVSHMEM_USE_NCCL=OFF \
    -DNVSHMEM_IBGDA_SUPPORT=OFF \
    -DNVSHMEM_IBRC_SUPPORT=OFF \
    -DNVSHMEM_PMIX_SUPPORT=OFF \
    -DNVSHMEM_TIMEOUT_DEVICE_POLLING=OFF \
    -DNVSHMEM_MPI_SUPPORT=ON \
    -DNVSHMEM_USE_GDRCOPY=ON \
    -DNVSHMEM_LIBFABRIC_SUPPORT=ON
    
cd build
make -j$(nproc)
sudo make install

nvshmem-info -a # Should display details of nvshmem
