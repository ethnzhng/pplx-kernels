#!/usr/bin/env bash
set -e

HOSTFILE="./hostfile"
HOSTS=($(awk '{print $1}' "$HOSTFILE"))
if [ ${#HOSTS[@]} -lt 2 ]; then
    echo "Error: Not enough hosts in the hostfile"
    exit 1
fi
SOURCE_HOST="${HOSTS[0]}"
TARGET_HOST="${HOSTS[1]}"
USER="ubuntu"

WORKDIR="$(realpath ..)/"
echo "Syncing $WORKDIR from $SOURCE_HOST to $TARGET_HOST..."
# chmod +x *.sh
rsync \
    -a \
    --progress \
    --exclude=".git" \
    --filter=":- .gitignore" \
    -e ssh \
    $WORKDIR $USER@$TARGET_HOST:$WORKDIR
echo "Workdirs synced"

REBUILD_PPLX=$1
if [ -n "$REBUILD_PPLX" ]; then
    echo "Rebuilding on both nodes"
    mpirun \
        -np 2 \
        -hostfile ${HOSTFILE} \
        -map-by ppr:1:node \
        --mca pml ^cm \
        --mca btl tcp,self \
        --mca btl_tcp_if_exclude lo,docker0 \
        --bind-to none \
        ./install-pplx.sh
else
    echo "Skipping rebuild"
fi
