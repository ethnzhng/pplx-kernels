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

mpirun \
    -n 16 \
    -tag-output \
    --map-by ppr:8:node \
    --hostfile ${HOSTFILE} \
    -x TIME_ID=${TIME_ID} \
    --mca pml ^cm \
    --mca btl tcp,self \
    --mca btl_tcp_if_exclude lo,docker0 \
    --bind-to none \
    ./profile-cpp-ata-ncu.sh \
    2>&1 | add_timestamp | tee $LOG_NAME
