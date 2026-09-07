#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
OLMOE=olmoe-1b-7b-0924-instruct-q4_k_m.gguf
OSS20=gpt-oss-20b-MXFP4.gguf

# --- olmoe: does llama-bench dedupe repeated values? does it abort on OOM?
run olmoe-threads "$OLMOE" -ncmoe 0 -t 16,4,8,12,16,20 -p 0 -n 64 -r 3
# --- olmoe: cost of offloading experts (16 blocks)
run olmoe-ncmoe "$OLMOE" -ncmoe 0,0,2,4,8,16 -t 16 -p 0 -n 64 -r 3
# --- gpt-oss-20b: descending ncmoe until OOM (24 blocks). Does csv survive the abort?
run oss20-ncmoe-desc "$OSS20" -ncmoe 12,12,10,8,6,4,2,0 -t 16 -p 0 -n 64 -r 3
