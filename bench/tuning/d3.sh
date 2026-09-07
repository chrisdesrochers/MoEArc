#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
Q30=Qwen3-30B-A3B-Q4_K_M.gguf
Q35=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
SCOUT=Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf
# Descending ncmoe: llama-bench streams csv per row and aborts at the first OOM,
# so one invocation yields both the throughput curve and the floor.
# First value repeated = discarded warm-up cell (same cell, no reload).
run qwen30-ncmoe "$Q30"   -ncmoe 30,30,28,26,24,22,20,19,18,17,16,14,12 -t 16 -p 0 -n 64 -r 3
run qwen35-ncmoe "$Q35"   -ncmoe 34,34,32,30,28,26,24,22,20,18,16,14,12 -t 16 -p 0 -n 64 -r 3
run scout-ncmoe  "$SCOUT" -ncmoe 48,48,46,44,42,40,38,36,34 -t 16 -p 0 -n 64 -r 3
