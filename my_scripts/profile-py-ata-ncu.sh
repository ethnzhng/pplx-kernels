#!/usr/bin/env bash
set -e

REPORT_NAME=${TIME_ID}_pplx_py_ata_node${NODE_RANK:-_single}

cd ..
mkdir -p reports
set -x

ncu \
    --export="reports/$REPORT_NAME" \
    --force-overwrite \
    --target-processes all \
    --replay-mode app-range \
    --nvtx \
    --nvtx-include "pplx_dispatch/" \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --section WorkloadDistribution \
    --section SchedulerStats \
    --section Nvlink \
    --section Nvlink_Tables \
    --section Nvlink_Topology \
    --section MemoryWorkloadAnalysis \
    python3 -m tests.profile_all_to_all --kernel="pplx_dispatch"

    # --import-source yes \
    # --set basic \

    # --metrics sm__warps_active.avg.per_cycle \
    # --kernel-name regex:"dispatchKernel" \
    # --replay-mode kernel \
    