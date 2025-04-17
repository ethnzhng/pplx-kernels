#!/usr/bin/env bash
set -e

# extract hosts from hostfile
HOSTFILE="./hostfile"
HOSTS=($(awk '{print $1}' "$HOSTFILE"))
if [ ${#HOSTS[@]} -lt 2 ]; then
    echo "Error: Not enough hosts in the hostfile"
    exit 1
fi
MASTER_HOST="${HOSTS[0]}"
MINION_HOST="${HOSTS[1]}"

# prepare log
export TIME_ID=$(date +"%Y%m%d%H%M%S")
mkdir -p ../logs
LOG_NAME="../logs/${TIME_ID}_pplx.log"

add_timestamp() {
    while IFS= read -r line; do
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $line"
    done
}

# run desired script on both nodes
SCRIPT_TO_RUN="./profile-bench-ata.sh"
mpirun \
    -n 2 \
    -tag-output \
    --map-by ppr:1:node \
    --hostfile ${HOSTFILE} \
    -x TIME_ID=${TIME_ID} \
    -x MASTER_HOST=${MASTER_HOST} \
    -x MINION_HOST=${MINION_HOST} \
    --mca pml ^cm \
    --mca btl tcp,self \
    --mca btl_tcp_if_exclude lo,docker0 \
    --bind-to none \
    ./multinode-wrapper.sh bash -c ${SCRIPT_TO_RUN} \
    2>&1 | add_timestamp | tee $LOG_NAME
