#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
OSS120=gpt-oss-120b-MXFP4.gguf
# What the llama-bench default (-t 4) actually costs on the flagship, measured WARM.
# The 2.1x figure in the repo came from runs at load 3.14-7.75; this replaces it.
run oss120-primer2  "$OSS120" -ncmoe 36 -t 16 -p 0 -n 32 -r 2
run oss120-default  "$OSS120" -ncmoe 36 -t 16,4,16,4,16 -p 0 -n 64 -r 3
echo D10-DONE
