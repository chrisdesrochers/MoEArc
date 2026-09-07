#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
M=olmoe-1b-7b-0924-instruct-q4_k_m.gguf
# smoke: does it load at ncmoe 0, and does csv stream incrementally?
run olmoe-smoke "$M" -ncmoe 0 -t 16 -p 0 -n 32 -r 2
