#!/usr/bin/env bash
set -e

# determine node rank based on current hostname
NORMALIZED_HOSTNAME=$(hostname | sed 's/^ip-//' | tr '-' '.')

if [ "$NORMALIZED_HOSTNAME" = "$MASTER_HOST" ]; then
    export NODE_RANK=0
elif [ "$NORMALIZED_HOSTNAME" = "$MINION_HOST" ]; then
    export NODE_RANK=1
else
    echo "Error: Unknown host or hostname"
    exit 1
fi

# set env vars required by multinode scripts
export WORLD_SIZE=16
export WORLD_LOCAL_SIZE=8
export MASTER_ADDR=$MASTER_HOST
export MASTER_PORT="29501"

# tell nvshmem to use efa
export NVSHMEM_REMOTE_TRANSPORT="libfabric"
export NVSHMEM_LIBFABRIC_PROVIDER="efa"

# export NCCL_DEBUG=INFO

# explicitly configure libfabric just in case
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA="1"

exec "$@"
