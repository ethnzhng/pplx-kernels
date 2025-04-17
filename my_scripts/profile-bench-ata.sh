#!/usr/bin/env bash
set -e

REPORT_NAME=${TIME_ID}_pplx_ata_node${NODE_RANK:-_single}

cd ..
mkdir -p reports
set -x
nsys profile \
    --trace="cuda,nvtx,mpi" \
    --cuda-memory-usage=true \
    --show-output true \
    --stats=true \
    --output="reports/$REPORT_NAME" \
    --force-overwrite=true \
    python3 -m tests.profile_all_to_all
    # python3 -m tests.bench_all_to_all

# if tracing orst then put
# --osrt-threshold 10000
# or higher num
