#!/bin/bash
# Run StereoAnywhere inference with MapReduce tiling.

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <stereo_ckpt> <mono_ckpt> <left_glob> <right_glob> [outdir]" >&2
  exit 1
fi

STEREO_CKPT=$1
MONO_CKPT=$2
LEFT_PATTERN=$3
RIGHT_PATTERN=$4
OUTDIR=${5:-"demo_output_mapreduce_v2"}

python demo/fast_demo_mapreduce_v2.py \
  --left ${LEFT_PATTERN} \
  --right ${RIGHT_PATTERN} \
  --loadstereomodel "${STEREO_CKPT}" \
  --loadmonomodel "${MONO_CKPT}" \
  --outdir "${OUTDIR}" \
  --mixed_precision \
  --clear_cache \
  --use_aggregate_mono_vol \
  --use_truncate_vol \
  --non_lambertian
