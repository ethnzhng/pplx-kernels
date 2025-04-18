#!/usr/bin/env bash
set -e

REPORT_NAME=${TIME_ID}_pplx_ata_node${NODE_RANK:-_single}

# https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering

cd ..
mkdir -p reports
set -x
# need to use app-range for distributed/collective ops
# only profile local rank 0
ncu \
    --export="reports/$REPORT_NAME" \
    --force-overwrite \
    --target-processes all \
    --replay-mode app-range \
    --nvtx \
    --nvtx-include "pplx_dispatch/" \
    --set basic \
    python3 -m tests.profile_all_to_all --kernel="pplx_dispatch"

    # --metrics sm__warps_active.avg.per_cycle
    # --replay-mode kernel \
    # --kernel-name regex:"^dispatchKernel$" \
    