#!/usr/bin/env bash
# Train several run configs at the same time on one GPU, splitting the CPU cores
# between their data loaders (the CPU, not the GPU, limits speed here).
# Usage (from the repository root, after scripts/pod/setup.sh):
#   bash scripts/pod/run_parallel.sh configs/a.yaml configs/b.yaml configs/c.yaml
# Set CPUS to the pod's allocated vCPU count (for example CPUS=16): inside a container,
# nproc can report the whole host (128 on one pod) instead of the allocation.
# Each run writes runs/<run_name>/ and logs/<run_name>.log. Safe to rerun: --resume
# continues each run from its last finished epoch. Detached with nohup, so closing
# the browser terminal does not stop training; watch with: tail -f logs/*.log
set -euo pipefail
[ "$#" -ge 1 ] || { echo "give one or more run configs"; exit 1; }
mkdir -p logs
CORES=${CPUS:-$(nproc)}
if [ -z "${CPUS:-}" ] && [ "$CORES" -gt 32 ]; then
  echo "nproc reports $CORES cores, which is probably the host, not this pod; set CPUS (for example CPUS=16)"; exit 1
fi
WORKERS=$(( (CORES - 1) / $# ))
[ "$WORKERS" -ge 1 ] || WORKERS=1
echo "$# runs, $CORES cores, $WORKERS data-loader workers each"
for CONFIG in "$@"; do
  NAME=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CONFIG")
  nohup python scripts/train.py --config "$CONFIG" --data-config configs/data.yaml \
    --out "runs/$NAME" --resume --num-workers "$WORKERS" > "logs/$NAME.log" 2>&1 &
  echo "started $NAME (pid $!), log: logs/$NAME.log"
done
echo "all started; follow with: tail -f logs/*.log"
